import os
import sys

os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'utils' else script_dir
sys.path.insert(0, root_dir)

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from model import EmotionClassifier 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import ast
import json

MODEL_PATH = 'best_model.pth'
VAL_CSV = 'data/processed/val.csv'
MODEL_NAME = 'microsoft/deberta-v3-base'  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 32

EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

class EmotionDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        lbl_str = self.df.iloc[idx]['labels']
        
        try:
            label_data = ast.literal_eval(lbl_str)
            # 兼容软标签和硬标签
            if len(label_data) > 0 and isinstance(label_data[0], float):
                target = torch.tensor(label_data, dtype=torch.float32)
            else:
                target = torch.zeros(28)
                for i in label_data:
                    target[int(i)] = 1.0
        except:
            # 容错处理
            target = torch.zeros(28)
                
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=MAX_LEN,
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target
        }

def search_thresholds_old(preds, targets, low=0.1, high=0.9, step=0.01):
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

def main():
    print(f"开始重新评估 (Device: {DEVICE})...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return
    if not os.path.exists(VAL_CSV):
        print(f"错误：找不到验证集 {VAL_CSV}")
        return

    print("加载验证集...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_ds = EmotionDataset(VAL_CSV, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_labels_list = [ds['labels'] for ds in val_ds]
    val_pos_counts = np.array(all_labels_list).sum(axis=0)
    print(f"验证集加载完成，共 {len(val_ds)} 条样本。")

    print("加载模型权重...")
    model = EmotionClassifier(MODEL_NAME, num_labels=28).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载失败：{e}")
        return
    
    model.eval()

    print("正在运行推理...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            targets = batch['labels'].to(DEVICE)
            
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    preds_array = np.array(all_preds)
    targets_array = np.array(all_targets)
    print("推理完成。")

    print("\n" + "="*30)
    print("策略 1: 原始全局搜索 (Baseline)")
    print("="*30)
    thresh_old = search_thresholds_old(preds_array, targets_array)
    preds_bin_old = (preds_array > thresh_old).astype(int)
    f1_macro_old = f1_score(targets_array, preds_bin_old, average='macro', zero_division=0)
    f1_micro_old = f1_score(targets_array, preds_bin_old, average='micro', zero_division=0)
    print(f"Macro F1: {f1_macro_old:.4f} | Micro F1: {f1_micro_old:.4f}")

    print("\n" + "="*30)
    print("策略 2: 手动微调特定类别阈值")
    print("="*30)
    
    thresh_final = thresh_old.copy()
    
    # 定义需要调整的类别 (索引：新阈值)
    # 基于之前的分析：realization(22), disappointment(9), anger(2) 等阈值过高
    adjustments = {
        22: 0.45,  # realization
        9: 0.40,   # disappointment
        2: 0.60,   # anger
        25: 0.55,  # sadness
        6: 0.50,   # confusion
        5: 0.50,   # caring
        13: 0.45,  # excitement
        14: 0.45,  # fear
        23: 0.40,  # relief 
        19: 0.40   # nervousness
    }

    print(f"{'Class':<15} | {'Old Thresh':<10} | {'New Thresh':<10} | {'Status'}")
    print("-" * 55)
    
    for idx, new_t in adjustments.items():
        if idx < 28: # 防止索引越界
            old_t = thresh_final[idx]
            thresh_final[idx] = new_t
            name = EMOTION_NAMES[idx]
            print(f"{name:<15} | {old_t:.2f}       | {new_t:.2f}       | ✅ Adjusted")

    preds_bin_final = (preds_array > thresh_final).astype(int)
    f1_macro_final = f1_score(targets_array, preds_bin_final, average='macro', zero_division=0)
    f1_micro_final = f1_score(targets_array, preds_bin_final, average='micro', zero_division=0)
    
    print(f"\nMacro F1: {f1_macro_final:.4f} | Micro F1: {f1_micro_final:.4f}")
    
    diff = f1_macro_final - f1_macro_old
    if diff > 0:
        print(f"相比基准提升: +{diff:.4f}")
    else:
        print(f"相比基准变化: {diff:.4f} (可能无需调整)")

   
    print("\n" + "="*30)
    print("各类别 F1 详情对比")
    print("="*30)
    print(f"{'Class':<15} | {'Count':<5} | {'Base F1':<8} | {'Tuned F1':<8} | {'Diff':<8}")
    print("-" * 65)
    
    f1_per_class_old = f1_score(targets_array, preds_bin_old, average=None, zero_division=0)
    f1_per_class_final = f1_score(targets_array, preds_bin_final, average=None, zero_division=0)
    
    improvements = []
    for i in range(28):
        name = EMOTION_NAMES[i]
        count = int(val_pos_counts[i])
        f1_old = f1_per_class_old[i]
        f1_new = f1_per_class_final[i]
        d = f1_new - f1_old
        improvements.append((name, count, f1_old, f1_new, d))
    
    # 按提升幅度排序显示
    improvements.sort(key=lambda x: x[4], reverse=True)
    
    for name, count, f1_old, f1_new, d in improvements:
        print(f"{name:<15} | {count:<5} | {f1_old:.4f}   | {f1_new:.4f}   | {d:+.4f}")

    # --- 保存最佳结果 ---
    if f1_macro_final >= f1_macro_old:
        result_data = {
            "macro_f1": float(f1_macro_final),
            "micro_f1": float(f1_micro_final),
            "strategy": "hybrid_manual_tuning",
            "thresholds": thresh_final.tolist(),
            "thresholds_named": {EMOTION_NAMES[i]: float(thresh_final[i]) for i in range(28)}
        }
        output_dir = os.path.join(root_dir, 'results', 'best_thresholds')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'best_thresholds_final.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n最佳阈值已保存至: {output_file}")
        print("提示：在提交测试结果时，请使用此文件中的 thresholds。")
    else:
        print(f"\n提示:原始阈值策略效果更好,建议使用原始阈值。")

if __name__ == "__main__":
    main()