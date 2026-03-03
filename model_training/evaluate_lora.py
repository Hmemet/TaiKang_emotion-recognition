import os
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import torch
import pandas as pd
import numpy as np
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'microsoft/deberta-v3-base'
MODEL_PATH = 'best_model_lora.pth'
VAL_CSV = 'data/processed/val.csv'
BATCH_SIZE = 32
MAX_LEN = 128
PICTURE_OUTPUT_ROOT = os.path.join('pictures')
RESULTS_OUTPUT_ROOT = os.path.join('results')

EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]


def get_picture_path(subfolder, filename):
    output_dir = os.path.join(PICTURE_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def get_results_path(subfolder, filename):
    output_dir = os.path.join(RESULTS_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

class SimpleDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
    
    def __len__(self): 
        return len(self.df)
    
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
        except Exception:
            pass
        
        enc = self.tokenizer(
            text, 
            max_length=MAX_LEN, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(), 
            'attention_mask': enc['attention_mask'].flatten(), 
            'labels': target
        }

def plot_metrics(f1_scores, best_thresh, names):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sorted_idx = np.argsort(f1_scores)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_scores = [f1_scores[i] for i in sorted_idx]
    sorted_thresh = [best_thresh[i] for i in sorted_idx]

    colors = ['#d73027' if s < 0.3 else '#fee08b' if s < 0.6 else '#1a9850' for s in sorted_scores]
    sns.barplot(x=sorted_scores, y=sorted_names, ax=axes[0], palette=colors, orient='h')
    axes[0].axvline(x=0.65, color='blue', linestyle='--', linewidth=2, label='Target 0.65')
    axes[0].set_title(f'Per-Class F1 Score (Macro Avg: {np.mean(f1_scores):.4f})', fontsize=14)
    axes[0].set_xlabel('F1-Score')
    axes[0].legend()

    sns.scatterplot(x=sorted_thresh, y=sorted_names, ax=axes[1], color='coral', s=100, zorder=3)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', label='Default 0.5')
    axes[1].set_title('Optimized Threshold per Class', fontsize=14)
    axes[1].set_xlabel('Threshold')
    axes[1].legend()

    plt.tight_layout()
    out_path = get_picture_path('evaluate_visuals', 'evaluation_visuals_lora.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"可视化图表已保存为 {out_path}")


def chunk_indices(total, chunk_size):
    return [list(range(i, min(i + chunk_size, total))) for i in range(0, total, chunk_size)]


def plot_label_correlation_heatmap(y_true, names):
    corr_df = pd.DataFrame(y_true, columns=names).corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_df,
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Label Correlation Heatmap (Ground Truth)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_path = get_picture_path('label_correlation_heatmap', 'label_correlation_heatmap_lora.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"标签相关性热力图已保存为 {out_path}")


def plot_precision_recall_curves(y_true, y_probs, names, group_size=7):
    groups = chunk_indices(len(names), group_size)
    micro_ap = average_precision_score(y_true.ravel(), y_probs.ravel())

    for group_idx, class_indices in enumerate(groups, 1):
        plt.figure(figsize=(10, 8))
        for i in class_indices:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            ap = average_precision_score(y_true[:, i], y_probs[:, i])
            plt.plot(recall, precision, linewidth=2, label=f"{names[i]} (AP={ap:.3f})")

        precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_probs.ravel())
        plt.plot(
            recall_micro,
            precision_micro,
            linestyle='--',
            color='black',
            linewidth=2,
            label=f"micro-average (AP={micro_ap:.3f})"
        )

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves (Group {group_idx})')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()

        out_path = get_picture_path(
            'precision_recall_curve_group',
            f'precision_recall_curve_group_{group_idx}_lora.png'
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"PR 曲线图已保存为 {out_path}")


def plot_confidence_distribution(y_probs, y_true):
    positive_conf = y_probs[y_true == 1]
    negative_conf = y_probs[y_true == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(negative_conf, bins=40, alpha=0.6, label='True Negative Labels', density=True)
    plt.hist(positive_conf, bins=40, alpha=0.6, label='True Positive Labels', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Confidence Distribution Histogram')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = get_picture_path('confidence_distribution_histogram', 'confidence_distribution_histogram_lora.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"置信度分布直方图已保存为 {out_path}")


def plot_grouped_radar_chart(report_data, group_size=7):
    precision_vals = [row['Precision'] for row in report_data]
    recall_vals = [row['Recall'] for row in report_data]
    f1_vals = [row['F1-Score'] for row in report_data]
    class_names = [row['Class'] for row in report_data]

    groups = chunk_indices(len(class_names), group_size)

    for group_idx, indices in enumerate(groups, 1):
        group_names = [class_names[i] for i in indices]
        group_precision = [precision_vals[i] for i in indices]
        group_recall = [recall_vals[i] for i in indices]
        group_f1 = [f1_vals[i] for i in indices]

        angles = np.linspace(0, 2 * np.pi, len(group_names), endpoint=False).tolist()
        angles += angles[:1]

        group_precision += group_precision[:1]
        group_recall += group_recall[:1]
        group_f1 += group_f1[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.plot(angles, group_precision, linewidth=2, label='Precision')
        ax.fill(angles, group_precision, alpha=0.15)
        ax.plot(angles, group_recall, linewidth=2, label='Recall')
        ax.fill(angles, group_recall, alpha=0.15)
        ax.plot(angles, group_f1, linewidth=2, label='F1-Score')
        ax.fill(angles, group_f1, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(group_names, fontsize=9)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)
        ax.set_title(f'Radar Chart: Precision/Recall/F1 (Group {group_idx})', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))

        plt.tight_layout()
        out_path = get_picture_path('radar_chart_group', f'radar_chart_group_{group_idx}_lora.png')
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"雷达图已保存为 {out_path}")

def run_evaluation():
    print(f"正在加载 LoRA 模型：{MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=28, 
        problem_type="multi_label_classification"
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(base_model, lora_config)
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return
        
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print("模型加载成功")

    val_ds = SimpleDataset(VAL_CSV, tokenizer)
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    y_probs, y_true = [], []
    print("正在对验证集进行预测...")
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

    print("正在为当前模型重新搜索最佳阈值 (步长 0.01)...")
    best_thresh = np.ones(28) * 0.5
    for i in range(28):
        best_f1_c = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.01): 
            score = f1_score(y_true[:, i], y_probs[:, i] > t, zero_division=0)
            if score > best_f1_c:
                best_f1_c = score
                best_t = t
        best_thresh[i] = best_t
    
    print(f"搜索完成，新阈值范围：[{best_thresh.min():.3f}, {best_thresh.max():.3f}]")

    y_pred = (y_probs > best_thresh).astype(int)
    
    overall_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    overall_f1 = np.mean([f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(28)])
    overall_micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    overall_accuracy = accuracy_score(y_true, y_pred) 
    
    report_data = []
    
    print("\n" + "="*70)
    print(f"最终 Macro F1 Score:   {overall_f1:.4f}")
    print(f"最终 Micro F1 Score:   {overall_micro_f1:.4f}")
    print(f"总体 Macro Precision:  {overall_precision:.4f}")
    print(f"总体 Macro Recall:     {overall_recall:.4f}")
    print(f"总体 Subset Accuracy:  {overall_accuracy:.4f} (所有标签完全匹配)")
    print("="*70)
    
    print("\n各类别详细报告:")
    print(f"{'Class':<15} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Acc@Class':<8} | {'Conf':<6}")
    print("-" * 70)

    for i, name in enumerate(EMOTION_NAMES):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

        class_correct = np.sum(y_true[:, i] == y_pred[:, i])
        class_acc = class_correct / len(y_true)
        
        positive_preds_mask = (y_pred[:, i] == 1)
        if np.sum(positive_preds_mask) > 0:
            avg_confidence = np.mean(y_probs[positive_preds_mask, i])
        else:
            avg_confidence = 0.0
            
        report_data.append({
            'Class': name,
            'Precision': p,
            'Recall': r,
            'F1-Score': f1,
            'Class_Accuracy': class_acc, 
            'Threshold': best_thresh[i],
            'Avg_Confidence_Positive': avg_confidence,
            'Support': int(np.sum(y_true[:, i]))
        })
        
        print(f"{name:<15} | {p:<6.4f} | {r:<6.4f} | {f1:<6.4f} | {class_acc:<8.4f} | {avg_confidence:<6.4f}")


    df_report = pd.DataFrame(report_data)
    
    overall_row = pd.DataFrame([{
        'Class': '** OVERALL (MACRO) **',
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1-Score': overall_f1,
        'Class_Accuracy': '-', 
        'Threshold': '-',
        'Avg_Confidence_Positive': '-',
        'Support': '-'
    }])
    
    micro_row = pd.DataFrame([{
        'Class': '** OVERALL (MICRO) **',
        'Precision': '-', 
        'Recall': '-',
        'F1-Score': overall_micro_f1,
        'Class_Accuracy': '-',
        'Threshold': '-',
        'Avg_Confidence_Positive': '-',
        'Support': '-'
    }])

    subset_acc_row = pd.DataFrame([{
        'Class': '** OVERALL (SUBSET ACC) **',
        'Precision': '-',
        'Recall': '-',
        'F1-Score': '-',
        'Class_Accuracy': overall_accuracy, 
        'Threshold': '-',
        'Avg_Confidence_Positive': '-',
        'Support': '-'
    }])

    final_df = pd.concat([overall_row, micro_row, subset_acc_row, df_report], ignore_index=True)
    
    report_path = get_results_path('final_report', 'final_report_lora.csv')
    final_df.to_csv(report_path, index=False, float_format='%.4f')
    print(f"\n详细报告已保存为 {report_path}")
    print("包含:Precision, Recall, F1, Class_Accuracy, Confidence, Support")

    plot_metrics([row['F1-Score'] for row in report_data], best_thresh, EMOTION_NAMES)
    plot_label_correlation_heatmap(y_true, EMOTION_NAMES)
    plot_precision_recall_curves(y_true, y_probs, EMOTION_NAMES, group_size=7)
    plot_confidence_distribution(y_probs, y_true)
    plot_grouped_radar_chart(report_data, group_size=7)
    
    threshold_path = get_results_path('best_thresholds', 'final_best_thresholds_verified.json')
    with open(threshold_path, 'w', encoding='utf-8') as f:
        json.dump({'thresholds': best_thresh.tolist(), 'macro_f1': overall_f1}, f, indent=2)
    print(f"验证后的最佳阈值已保存为 {threshold_path}")

if __name__ == "__main__":
    run_evaluation()