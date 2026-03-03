import os
os.environ['HF_HOME'] = 'C:/hf_cache'  # 使用英文路径
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import torch
import torch.nn as nn
import pandas as pd
import ast
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm
from model import EmotionClassifier
import json
from torch.amp import autocast, GradScaler

from peft import LoraConfig, get_peft_model, TaskType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#核心配置区
MODEL_NAME = 'microsoft/deberta-v3-base' # 尝试改为 'microsoft/deberta-v3-large' 以利用 LoRA 省下的显存！
MAX_LEN = 128
BATCH_SIZE = 16
WEIGHT_DECAY = 0.05
DEFAULT_EPOCHS = 16
LR = 2e-5
ACCUM_STEPS = 2
DEFAULT_GAMMA = 1.0


USE_LORA = False  #设为 True 开启 LoRA，False 关闭
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# 采样与少类增强相关配置
SMALL_CLASS_PERCENTILE = float(os.environ.get('SMALL_CLASS_PERCENTILE', '25'))
TRAIN_CSV = 'data/processed/train_soft_augmented.csv'
VAL_CSV = 'data/processed/val.csv'
# 

RESULTS_OUTPUT_ROOT = os.path.join('results')

def get_results_path(subfolder, filename):
    output_dir = os.path.join(RESULTS_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

class EmotionDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        target = torch.zeros(28)
        try:
            labels_data = ast.literal_eval(self.df.iloc[idx]['labels'])
            # 兼容软标签和硬标签
            if len(labels_data) > 0 and isinstance(labels_data[0], float):
                for i, val in enumerate(labels_data):
                    if i < 28: target[i] = val
            else:
                # 硬标签
                for i in labels_data:
                    if i < 28: target[int(i)] = 1.0
        except:
            pass
            
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=MAX_LEN,
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target
        }


def get_pos_weights(dataset):
    print("正在计算类别权重...")
    all_labels = [dataset[i]['labels'].numpy() for i in range(len(dataset))]
    # 对于软标签，这里统计的是概率和，近似当作计数
    pos_counts = np.array(all_labels).sum(axis=0)
    neg_counts = len(dataset) - pos_counts

    pos_weights = neg_counts / (pos_counts + 1e-5)
    pos_weights = np.sqrt(pos_weights)
    pos_weights = np.clip(pos_weights, 1.0, 12.0)

    try:
        threshold_count = np.percentile(pos_counts, SMALL_CLASS_PERCENTILE)
        rare_mask = pos_counts < threshold_count
        current_boost = float(os.environ.get('SMALL_CLASS_BOOST', '1.0'))
        
        if rare_mask.any() and current_boost != 1.0:
            pos_weights[rare_mask] = pos_weights[rare_mask] * current_boost
            pos_weights = np.clip(pos_weights, 1.0, 12.0)
            print(f"Boosted pos_weights for {int(rare_mask.sum())} rare classes with factor {current_boost}")
    except Exception as e:
        print(f"计算 Boost 权重时出错: {e}")

    print("pos_counts (approx):", np.round(pos_counts, 1).tolist())
    print("pos_weights:", np.round(pos_weights, 3).tolist())

    return torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)


def compute_sample_weights(dataset, pos_weights):
    labels_list = [dataset[i]['labels'].numpy() for i in range(len(dataset))]
    all_labels = np.array(labels_list)
    pos_counts = all_labels.sum(axis=0)
    eps = 1e-6
    class_weights = 1.0 / (pos_counts + eps)
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.8, 6.0)

    weights = []
    for lbl in labels_list:
        idx = np.where(lbl > 0.5)[0] # 软标签下，大于 0.5 视为正类
        if len(idx) == 0:
            w = 1.0
        else:
            w = float(class_weights[idx].sum())
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def search_thresholds(preds, targets, low=0.1, high=0.9, step=0.01):
    n_classes = preds.shape[1]
    thresholds = np.full(n_classes, 0.5)
    for i in range(n_classes):
        best_t = 0.5
        best_f1 = 0.0
        t = low
        while t <= high:
            score = f1_score(targets[:, i], preds[:, i] > t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
            t = round(t + step, 6)
        thresholds[i] = best_t
    return thresholds


def train(focal_gamma=None, epochs_override=None, use_subset=False, subset_size=2000, 
          sampler_scale=None, small_class_boost=None, dropout_rate=0.2):
    
    GAMMA = focal_gamma if focal_gamma is not None else DEFAULT_GAMMA
    EPOCHS = epochs_override if epochs_override is not None else DEFAULT_EPOCHS
    
    # 处理 Sampler 参数
    SAMPLER_SCALE = sampler_scale if sampler_scale is not None else float(os.environ.get('SAMPLER_SCALE', '1.2'))
    SMALL_CLASS_BOOST = small_class_boost if small_class_boost is not None else float(os.environ.get('SMALL_CLASS_BOOST', '1.2'))
    os.environ['SAMPLER_SCALE'] = str(SAMPLER_SCALE)
    os.environ['SMALL_CLASS_BOOST'] = str(SMALL_CLASS_BOOST)

    print(f"当前配置: Gamma={GAMMA}, Epochs={EPOCHS}, Subset={use_subset}")
    print(f"LoRA={USE_LORA}, Model={MODEL_NAME}")
    if USE_LORA:
        print(f"LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = EmotionDataset(TRAIN_CSV, tokenizer)
    val_ds = EmotionDataset(VAL_CSV, tokenizer)

    if use_subset:
        print(f"[短跑模式] 激活！仅使用前 {subset_size} 条训练数据...")
        original_len = len(train_ds.df)
        train_ds.df = train_ds.df.iloc[:subset_size].reset_index(drop=True)
        print(f"原始数据量: {original_len} -> 截断后: {len(train_ds.df)}")
    else:
        print("[全量模式] 使用全部训练数据。")
    
    pos_weights = get_pos_weights(train_ds)

    use_sampler = os.environ.get('USE_SAMPLER', '0') == '1' # 默认关闭 Sampler，避免与 LoRA 冲突或调试复杂化
    if use_sampler:
        sample_weights = compute_sample_weights(train_ds, pos_weights)
        try:
            sample_weights = sample_weights * SAMPLER_SCALE
            sample_weights = sample_weights / sample_weights.mean()
        except:
            pass
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    #模型加载核心逻辑
    if USE_LORA:
        print("加载 LoRA 模型...")
        #直接使用 HF 的原生分类模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=28,
            problem_type="multi_label_classification"
        )
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["query_proj", "value_proj"], # DeBERTa v3 的关键层
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        model.to(DEVICE)
        model.float()
        
    else:
        print("加载自定义模型 (EmotionClassifier)...")
        # 非 LoRA 模式，使用自定义类
        model = EmotionClassifier(MODEL_NAME, 28, dropout_rate=dropout_rate).to(DEVICE)
        model.float()

    criterion = FocalLoss(pos_weight=pos_weights, gamma=GAMMA, reduction='mean')

    # 检查续训
    RESUME_FROM_CHECKPOINT = True
    checkpoint_path = 'best_model.pth'
    
    if USE_LORA:
        checkpoint_path = 'best_model_lora.pth'

    if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        print(f"正在加载已有权重: {checkpoint_path}")
        # 注意：LoRA 模型加载需要严格匹配结构
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("未找到权重或选择不加载，将从头训练。")

    # 优化器设置
    if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        LR_FINETUNE = LR * 0.2
        WARMUP_RATIO = 0.05
        print(f"续训模式：LR 调整为 {LR_FINETUNE}")
        optimizer = AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        WARMUP_RATIO = 0.1
   
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    scaler = GradScaler()


    best_f1 = 0.0
    patience = 2
    min_delta = 1e-4
    no_improve_epochs = 0
    lr_reduced = False
    stop_training = False

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        
        for step, batch in enumerate(loop):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['labels'].to(DEVICE)

            with autocast('cuda'):
                outputs = model(ids, mask)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                loss = criterion(logits, targets)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        if (step + 1) % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # 验证
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                outputs = model(ids, mask)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(batch['labels'].numpy())

        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        max_prob = preds_array.max()

        thresholds = search_thresholds(preds_array, targets_array, low=0.1, high=0.9, step=0.01)
        
        # 保存阈值
        try:
            threshold_path = get_results_path('best_thresholds', 'best_thresholds.json')
            with open(threshold_path, 'w') as f:
                json.dump({'thresholds': thresholds.tolist()}, f, indent=2)
        except Exception as e:
            print(f"保存阈值时出错: {e}")

        preds_bin = (preds_array > thresholds).astype(int)
        best_val_f1 = f1_score(targets_array, preds_bin, average='macro', zero_division=0)
        micro_f1 = f1_score(targets_array, preds_bin, average='micro', zero_division=0)
        per_class_f1 = f1_score(targets_array, preds_bin, average=None, zero_division=0)
        
        try:
            per_class_f1_list = [float(round(x, 4)) for x in per_class_f1]
        except:
            per_class_f1_list = per_class_f1.tolist()

        print("Per-class F1:", per_class_f1_list)     
        print(f"Epoch {epoch + 1} | Macro F1: {best_val_f1:.4f} | Micro F1: {micro_f1:.4f} | Max Prob: {max_prob:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # 早停与保存
        if (best_val_f1 > best_f1 + min_delta) and not np.isnan(max_prob):
            best_f1 = best_val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存最佳模型 (F1={best_f1:.4f}) 到 {checkpoint_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            if not lr_reduced:
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                print("降低学习率并继续...")
                lr_reduced = True
                no_improve_epochs = 0
            else:
                print("训练结束。")
                stop_training = True
        
        if stop_training:
            break
            
    return best_f1

if __name__ == "__main__":
    train()