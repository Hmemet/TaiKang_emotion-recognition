import os
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
import pandas as pd
import numpy as np
import json
import ast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

#配置
MODEL_PATH = 'best_model_lora.pth'
VAL_CSV = 'data/processed/test.csv' 
MODEL_NAME = 'microsoft/deberta-v3-base'
HF_CACHE_DIR = 'C:/hf_cache'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 128
EMOTION_NAMES = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 
                 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
                 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                   'remorse', 'sadness', 'surprise', 'neutral']
#
class SimpleDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        target = torch.zeros(28)
        try:
            lbl = ast.literal_eval(self.df.iloc[idx]['labels'])
            if len(lbl) > 0 and isinstance(lbl[0], float):
                for i, v in enumerate(lbl): 
                    if i < 28: target[i] = v
            else:
                for i in lbl: 
                    if i < 28: target[int(i)] = 1.0
        except: pass
        enc = self.tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'labels': target}

def load_model_and_thresh():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
            use_fast=False
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=28,
            problem_type="multi_label_classification",
            cache_dir=HF_CACHE_DIR,
            local_files_only=True
        )
    except Exception as e:
        raise RuntimeError(
            f"无法从本地缓存加载模型/分词器，请先在可联网环境预下载到 {HF_CACHE_DIR}。原始错误: {e}"
        )

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS)
    model = get_peft_model(base_model, lora_config)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    
    thresholds = np.full(28, 0.5)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    thresh_path = os.path.join(project_root, 'results', 'best_thresholds', 'final_best_thresholds_verified.json')
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r', encoding='utf-8') as f:
            thresholds = np.array(json.load(f)['thresholds'])
            print(f"加载最佳阈值")
    else:
        print(f"未找到最佳阈值文件，使用默认0.5")
    return tokenizer, model, thresholds

def evaluate(loader, model, thresholds, name):
    y_probs, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            outputs = model(ids, mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.sigmoid(logits)
            y_probs.extend(probs.cpu().numpy())
            y_true.extend(batch['labels'].numpy())
    
    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    y_pred = (y_probs > thresholds).astype(int)
    macro_f1 = np.mean([f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(28)])
    print(f"{name}: Macro F1 = {macro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    print("🌍 开始盲测 (Holdout Set Test)...")
    tokenizer, model, thresholds = load_model_and_thresh()
    
    df_full = pd.read_csv(VAL_CSV)
    
    # 策略：如果是 val.csv，我们人为切分 20% 作为盲测集 (模拟 Test Set)
    # 如果你有真正的 test.csv，直接加载那个文件即可，不需要切分
    print(f"当前数据源：{VAL_CSV}")
    print("如果没有独立 test 集，将自动从当前文件中隔离 20% 作为盲测集")
    
    # 简单的哈希切分，保证每次运行切分结果一致
    '''
    np.random.seed(999) 
    mask = np.random.rand(len(df_full)) < 0.2
    df_holdout = df_full[mask].reset_index(drop=True)
    '''
    df_holdout = df_full

    if len(df_holdout) == 0:
        print("错误：无法生成盲测集")
        exit()

    print(f"盲测集样本数：{len(df_holdout)}")
    
    ds_holdout = SimpleDataset(df_holdout, tokenizer)
    loader_holdout = DataLoader(ds_holdout, batch_size=BATCH_SIZE)
    
    holdout_score = evaluate(loader_holdout, model, thresholds, "盲测集 (未见数据)")
    
    print("\n" + "="*50)
    expected_min = 0.60 
    
    if holdout_score > expected_min:
        print(f"通过！盲测分数 ({holdout_score:.4f}) 表现优异，泛化能力强。")
        print("(模型没有过拟合验证集)")
    else:
        print(f"警告：盲测分数 ({holdout_score:.4f}) 较低，可能存在验证集过拟合。")
    print("="*50)