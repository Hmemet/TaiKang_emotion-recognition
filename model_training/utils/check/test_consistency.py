import os
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import torch
import pandas as pd
import numpy as np
import json
import ast
import random
import string
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

#配置
MODEL_PATH = 'best_model_lora.pth'
VAL_CSV = 'data/processed/val.csv'
MODEL_NAME = 'microsoft/deberta-v3-base'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 128
EMOTION_NAMES = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                  'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                      'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# 噪声生成函数
def add_typo(text, prob=0.1):
    """随机制造拼写错误 (交换相邻字符或删除字符)"""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob and chars[i].isalpha():
            if i > 0 and random.random() > 0.5:
                chars[i], chars[i-1] = chars[i-1], chars[i] # 交换
            else:
                chars[i] = '' # 删除
    return "".join(chars)

def replace_synonym(text):
    # 简单同义词替换
    # 实际项目中可接入 WordNet，这里做简单演示
    replacements = {'good': 'great', 'bad': 'terrible', 'happy': 'glad', 'sad': 'unhappy', 'love': 'adore'}
    words = text.split()
    new_words = [replacements.get(w.lower(), w) for w in words]
    return " ".join(new_words)

def add_noise(text, mode='typo'):
    if mode == 'typo': return add_typo(text, 0.05)
    if mode == 'synonym': return replace_synonym(text)
    if mode == 'upper': return text.upper() if random.random() > 0.5 else text.lower()
    return text

class NoiseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, noise_mode='typo'):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.noise_mode = noise_mode
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        # 应用噪声
        noisy_text = add_noise(text, self.noise_mode)
        
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
        
        enc = self.tokenizer(noisy_text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'labels': target}

def load_model_and_thresh():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=28, problem_type="multi_label_classification")
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
    print("开始鲁棒性压力测试...")
    tokenizer, model, thresholds = load_model_and_thresh()
    
    # 1. 基准测试 (无噪声)
    ds_clean = NoiseDataset(VAL_CSV, tokenizer, noise_mode='none')
    loader_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE)
    base_score = evaluate(loader_clean, model, thresholds, "✅ 基准 (无噪声)")
    
    modes = ['typo', 'synonym', 'upper']
    drop_rates = []
    
    for mode in modes:
        ds_noise = NoiseDataset(VAL_CSV, tokenizer, noise_mode=mode)
        loader_noise = DataLoader(ds_noise, batch_size=BATCH_SIZE)
        score = evaluate(loader_noise, model, thresholds, f"⚠️ 噪声测试 ({mode})")
        drop = base_score - score
        drop_rates.append(drop)
        status = "🟢 优秀" if drop < 0.02 else ("🟡 一般" if drop < 0.05 else "🔴 危险")
        print(f"   -> 分数下降: {drop:.4f} [{status}]")
    
    avg_drop = np.mean(drop_rates)
    print("\n" + "="*50)
    if avg_drop < 0.03:
        print("通过！模型鲁棒性极强，抗干扰能力优秀。")
    elif avg_drop < 0.06:
        print("警告：模型对噪声有一定敏感度，但在可接受范围内。")
    else:
        print("严重警告：模型过拟合严重，轻微噪声导致性能崩盘！")
    print("="*50)