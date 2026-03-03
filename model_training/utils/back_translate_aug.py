import os
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
import ast
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME_EN_DE = "Helsinki-NLP/opus-mt-en-de"
MODEL_NAME_DE_EN = "Helsinki-NLP/opus-mt-de-en"


def load_translator(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

def translate_batch(texts, tokenizer, model, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            translated_ids = model.generate(**inputs, max_length=128)
        batch_results = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        results.extend(batch_results)
        print(f"Translated {min(i+batch_size, len(texts))}/{len(texts)}")
    return results

def augment_minority_classes(input_csv, output_csv, top_k_minority=10, augment_factor=3):
    print("加载数据...")
    df = pd.read_csv(input_csv)
    
    # 解析标签并统计频率
    def get_labels(lbl_str):
        lst = ast.literal_eval(lbl_str)
        # 如果是软标签 (float)，取 > 0.1 的作为有效标签；如果是硬标签 (0/1)，取 == 1 的
        if isinstance(lst[0], float):
            return [i for i, v in enumerate(lst) if v > 0.1]
        else:
            return [i for i, v in enumerate(lst) if v == 1]

    label_counts = np.zeros(28)
    df['parsed_labels'] = df['labels'].apply(get_labels)
    
    for lbls in df['parsed_labels']:
        for l in lbls:
            label_counts[l] += 1
            
    # 找到样本最少的 top_k 个类别
    minority_classes = np.argsort(label_counts)[:top_k_minority]
    print(f"少数类 ID: {minority_classes} (Counts: {label_counts[minority_classes].astype(int)})")
    
    # 筛选出包含这些少数类的样本
    minority_mask = df['parsed_labels'].apply(lambda x: any(l in minority_classes for l in x))
    minority_df = df[minority_mask].copy()
    
    if len(minority_df) == 0:
        print("未找到少数类样本")
        return

    print(f"找到 {len(minority_df)} 条包含少数类的样本，准备回译...")
    
    # 加载模型
    print("加载翻译模型 (En->De)...")
    tok_en_de, model_en_de = load_translator(MODEL_NAME_EN_DE)
    print("加载翻译模型 (De->En)...")
    tok_de_en, model_de_en = load_translator(MODEL_NAME_DE_EN)
    
    augmented_rows = []
    
    # 对每个少数类样本进行回译
    # 为了多样性，我们可以多次回译 (这里简单起见，每条只回译一次，如需多次可循环)
    texts_to_translate = minority_df['text'].tolist()
    
    print("步骤 1: 英语 -> 德语")
    de_texts = translate_batch(texts_to_translate, tok_en_de, model_en_de)
    
    print("步骤 2: 德语 -> 英语 (回译)")
    en_back_texts = translate_batch(de_texts, tok_de_en, model_de_en)
    
    # 构建新数据行
    for idx, new_text in enumerate(en_back_texts):
        original_row = minority_df.iloc[idx]
        new_row = original_row.copy()
        new_row['text'] = new_text
        # 标签保持不变
        augmented_rows.append(new_row)
        
    # 合并数据
    aug_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f"原始数据: {len(df)}, 增强后: {len(final_df)}, 新增: {len(aug_df)}")
    final_df.drop(columns=['parsed_labels'], inplace=True, errors='ignore')
    final_df.to_csv(output_csv, index=False)
    print(f"增强数据已保存至: {output_csv}")

if __name__ == "__main__":
    # 请修改为你的实际路径
    augment_minority_classes(
        'data/processed/train_soft_corrected.csv', 
        'data/processed/train_soft_augmented.csv',
        top_k_minority=10, # 增强最少的 10 个类
        augment_factor=1   # 每个样本回译 1 次 
    )
