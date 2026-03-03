import os
os.environ['HF_HOME'] = 'C:/hf_cache'  # 使用英文路径
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer
from model import EmotionClassifier
from train import EmotionDataset

PICTURE_OUTPUT_ROOT = os.path.join('pictures')
RESULTS_OUTPUT_ROOT = os.path.join('results')


def get_picture_path(subfolder, filename):
    output_dir = os.path.join(PICTURE_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def get_results_path(subfolder, filename):
    output_dir = os.path.join(RESULTS_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'microsoft/deberta-v3-base'


def plot_metrics(final_report, best_thresh, names):
    f1_scores = [final_report[name]['f1-score'] for name in names]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.barplot(x=f1_scores, y=names, ax=axes[0], palette='viridis', orient='h')
    axes[0].axvline(x=0.65, color='red', linestyle='--', label='Target 0.65')
    axes[0].set_title('F1-Score per Emotion Class', fontsize=14)
    axes[0].set_xlabel('F1-Score')
    axes[0].legend()

    sns.scatterplot(x=best_thresh, y=names, ax=axes[1], color='coral', s=100)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', label='Default 0.5 Threshold')
    axes[1].set_title('Optimized Threshold per Emotion Class', fontsize=14)
    axes[1].set_xlabel('Best Threshold')
    axes[1].legend()

    plt.tight_layout()
    out_evaluate_visuals_path = get_picture_path('evaluate_visuals', 'evaluation_visuals.png')
    plt.savefig(out_evaluate_visuals_path, dpi=300)
    print(f"可视化图表已生成并保存为 {out_evaluate_visuals_path}")


def run_evaluation():
    with open('data/label_mapping.json', 'r') as f:
        label_map = json.load(f)
    names = [label_map[str(i)] for i in range(len(label_map))]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmotionClassifier(model_name=MODEL_NAME, num_labels=28).to(DEVICE)
    model.float()

    print("正在加载最佳模型权重...")
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()

    val_ds = EmotionDataset('data/processed/val.csv', tokenizer)
    loader = DataLoader(val_ds, batch_size=32)

    y_probs, y_true = [], []
    print("正在对验证集进行预测...")
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            y_probs.extend(torch.sigmoid(logits).cpu().numpy())
            y_true.extend(batch['labels'].numpy())

    y_probs, y_true = np.array(y_probs), np.array(y_true)

    print("正在搜索每个类别的最佳 F1 阈值...")
    best_thresh = np.ones(28) * 0.5
    for i in range(28):
        best_f1_c = 0
        for t in np.arange(0.1, 0.9, 0.05):
            score = f1_score(y_true[:, i], y_probs[:, i] > t, zero_division=0)
            if score > best_f1_c:
                best_f1_c = score
                best_thresh[i] = t

    y_pred = np.zeros_like(y_probs)
    for i in range(28):
        y_pred[:, i] = (y_probs[:, i] > best_thresh[i]).astype(int)

    final_report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)

    print(f"\nFinal Macro F1 Score: {final_report['macro avg']['f1-score']:.4f}\n")

    pd.DataFrame(final_report).transpose().to_csv(get_results_path('final_report', 'final_report.csv'), float_format='%.4f')

    plot_metrics(final_report, best_thresh, names)


if __name__ == "__main__":
    run_evaluation()