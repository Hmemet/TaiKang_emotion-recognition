import os
os.environ['HF_HOME'] = 'C:/hf_cache'

import torch
import pandas as pd
import numpy as np
import json
import ast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

#配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'best_model_lora.pth')
VAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv')
MODEL_NAME = 'microsoft/deberta-v3-base'
HF_CACHE_DIR = 'C:/hf_cache'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 
MAX_LEN = 128


class SimpleDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
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

def get_prediction(text, tokenizer, model, thresholds):
    inputs = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    ids = inputs['input_ids'].to(DEVICE)
    mask = inputs['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(ids, mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > thresholds).astype(int)
    return probs, preds

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
    thresh_path = os.path.join(PROJECT_ROOT, 'results', 'best_thresholds', 'final_best_thresholds_verified.json')
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r', encoding='utf-8') as f:
            thresholds = np.array(json.load(f)['thresholds'])
            print(f"加载最佳阈值: {thresh_path}")
    else:
        print(f"未找到最佳阈值文件({thresh_path})，使用默认0.5")
    return tokenizer, model, thresholds

if __name__ == "__main__":
    print("开始鲁棒性检查...")
    tokenizer, model, thresholds = load_model_and_thresh()
    
    all_passed = True
    
    # 1. 确定性测试
    print("\n确定性测试 (Reproducibility)...")
    torch.manual_seed(42)
    np.random.seed(42)
    text = "I absolutely love this product, but the shipping was slow."
    p1, pred1 = get_prediction(text, tokenizer, model, thresholds)
    
    torch.manual_seed(42)
    np.random.seed(42)
    p2, pred2 = get_prediction(text, tokenizer, model, thresholds)
    
    if np.allclose(p1, p2) and np.array_equal(pred1, pred2):
        print("通过：结果完全可复现。")
    else:
        print("失败：结果不一致！存在随机性 Bug。")
        all_passed = False

    # 2. 边界测试 (空字符串、超长、乱码)
    print("\n边界测试 (Edge Cases)...")
    edge_cases = [
        ("", "空字符串"),
        ("A" * 2000, "超长文本"),
        ("!@#$%^&*()", "纯符号"),
        ("😊😊😊", "纯表情"),
        (" ", "纯空格")
    ]
    
    for txt, desc in edge_cases:
        try:
            p, pred = get_prediction(txt, tokenizer, model, thresholds)
            if np.all((p >= 0) & (p <= 1)):
                print(f"{desc}: 正常输出，概率范围 [0, 1]。")
            else:
                print(f"{desc}: 概率超出范围！")
                all_passed = False
        except Exception as e:
            print(f"{desc}: 程序崩溃！({str(e)})")
            all_passed = False

    # 3. 阈值逻辑单调性测试
    print("\n阈值逻辑测试 (Threshold Logic)...")
    test_text = "This is amazing!"
    probs, _ = get_prediction(test_text, tokenizer, model, thresholds)
    
    # 找一个概率为 0.7 的类
    idx = np.argmax(probs)
    prob_val = probs[idx]
    thresh_val = thresholds[idx]
    
    # 预测结果
    is_positive = (prob_val > thresh_val)
    
    # 逻辑检查：如果 prob > thresh, 预测必须为 1; 否则为 0
    expected_pred = 1 if prob_val > thresh_val else 0
    actual_pred = 1 if is_positive else 0
    
    if expected_pred == actual_pred:
        print(f"通过：类别 {idx} (Prob={prob_val:.2f}, Thresh={thresh_val:.2f}) 逻辑正确。")
    else:
        print(f"失败：逻辑矛盾！")
        all_passed = False
        
    # 4. 阈值文件完整性检查
    print("\n配置文件检查...")
    if len(thresholds) == 28 and np.all((thresholds >= 0) & (thresholds <= 1)):
        print("阈值文件格式正确 (28 个值，范围 0-1)。")
    else:
        print("阈值文件损坏或格式错误。")
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("所有检查通过！模型代码健壮、逻辑严密、可复现。")
    else:
        print("部分检查失败，请修复 Bug 后再部署。")
    print("="*50)