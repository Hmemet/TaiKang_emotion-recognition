# save as: verify_shuffle.py
import sys
import os
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from peft import PeftModel
import ast
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'utils' else script_dir
sys.path.insert(0, root_dir)

#配置
MODEL_PATH = 'best_model_lora.pth'  # 你刚刚训练好的最佳模型
VAL_CSV = 'data/processed/val.csv'
MODEL_NAME = 'microsoft/deberta-v3-base'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 128
#

class SimpleDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        # 解析标签
        try:
            lbl = ast.literal_eval(self.df.iloc[idx]['labels'])
            target = torch.zeros(28)
            if len(lbl) > 0 and isinstance(lbl[0], float):
                for i, v in enumerate(lbl): 
                    if i < 28: target[i] = v
            else:
                for i in lbl: 
                    if i < 28: target[int(i)] = 1.0
        except: target = torch.zeros(28)
        
        enc = self.tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'labels': target}

print("正在加载模型和数据...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df = pd.read_csv(VAL_CSV)

# 1. 正常预测
ds_normal = SimpleDataset(df, tokenizer)
loader_normal = DataLoader(ds_normal, batch_size=BATCH_SIZE)

# 2. 打乱标签预测
df_shuffled = df.copy()
np.random.seed(42)
np.random.shuffle(df_shuffled['labels'].values) # 彻底打乱标签列
ds_shuffle = SimpleDataset(df_shuffled, tokenizer)
loader_shuffle = DataLoader(ds_shuffle, batch_size=BATCH_SIZE)

# 加载模型
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=28, problem_type="multi_label_classification")

from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS)
model = get_peft_model(base_model, lora_config)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE).eval()

def evaluate(loader, name):
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            outs = model(ids, mask)
            logits = outs.logits if hasattr(outs, 'logits') else outs
            probs = torch.sigmoid(logits)
            all_p.extend(probs.cpu().numpy())
            all_t.extend(batch['labels'].numpy())
    
    preds = np.array(all_p)
    targets = np.array(all_t)
    
    # 简单阈值 0.5 
    binary = (preds > 0.5).astype(int)
    macro = f1_score(targets, binary, average='macro', zero_division=0)
    micro = f1_score(targets, binary, average='micro', zero_division=0)
    print(f"{name}: Macro F1 = {macro:.4f}, Micro F1 = {micro:.4f}")
    return macro

print("\n--- 开始测试 ---")
f1_normal = evaluate(loader_normal, "正常验证集 (参考)")

f1_shuffle = evaluate(loader_shuffle, "标签打乱验证集 (测谎)")

print("\n--- 结论 ---")
if f1_shuffle < 0.05:
    print("验证通过！模型没有作弊，分数真实有效！")
elif f1_shuffle < 0.15:
    print("警告：打乱后分数略高，可能存在轻微的数据分布偏差，但大概率没问题。")
else:
    print("严重警告！打乱标签后分数依然很高，存在严重的数据泄露！请检查代码！")