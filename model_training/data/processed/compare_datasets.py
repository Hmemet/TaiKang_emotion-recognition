import pandas as pd
import numpy as np
import ast
import json


FILE_OLD = 'data/processed/train_augmented.csv'

FILE_NEW = 'data/processed/train_soft_augmented.csv' 
# FILE_NEW = 'data/processed/train_soft_corrected.csv' # 先只验证软标签合并

EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]


def analyze_file(filepath, label):
    print(f"\n{'='*20} {label} 分析 {'='*20}")
    try:
        df = pd.read_csv(filepath)
        print(f"总行数 (样本数): {len(df):,}")
        
        # 1. 检查标签格式 (软标签 vs 硬标签)
        sample_lbl_str = df.iloc[0]['labels']
        sample_lbl = ast.literal_eval(sample_lbl_str)
        
        is_soft = any(isinstance(x, float) and 0 < x < 1 for x in sample_lbl)
        is_hard = all(x in [0.0, 1.0, 0, 1] for x in sample_lbl)
        
        print(f"标签类型: {'软标签 (Soft)' if is_soft else '硬标签 (Hard)'}")
        if is_soft:
            # 计算平均每个样本有多少个非零标签
            avg_labels_per_sample = []
            for lbl_str in df['labels'].head(1000): # 采样前1000条
                lbl = ast.literal_eval(lbl_str)
                count = sum(1 for x in lbl if x > 0.1) # 阈值0.1视为有效标签
                avg_labels_per_sample.append(count)
            print(f"平均均每样本有效标签数: {np.mean(avg_labels_per_sample):.2f}")
            
            # 检查是否有真正的概率值 (非 0/1)
            has_prob = False
            for lbl_str in df['labels'].head(100):
                lbl = ast.literal_eval(lbl_str)
                if any(0 < x < 1 for x in lbl):
                    has_prob = True
                    break
            print(f"   检测到真实概率分布 (0<x<1): {'是' if has_prob else '否 (全是0或1)'}")

        # 2. 统计类别分布 (检查不平衡情况)
        class_counts = np.zeros(28)
        for lbl_str in df['labels']:
            lbl = ast.literal_eval(lbl_str)
            for i, val in enumerate(lbl):
                if val > 0.5: # 简单统计：大于0.5算作正样本 (对于软标签这是一种近似)
                    class_counts[i] += 1
        
        # 找出最少的 3 个类
        min_indices = np.argsort(class_counts)[:3]
        print(f"最少样本的 3 个类:")
        for idx in min_indices:
            print(f"   - {EMOTION_NAMES[idx]}: {int(class_counts[idx]):,} 条")
            
        return len(df), is_soft, class_counts

    except Exception as e:
        print(f"分析失败: {e}")
        return 0, False, None

def main():
    print("开始对比数据集质量...")
    
    old_count, old_is_soft, old_counts = analyze_file(FILE_OLD, "旧数据集 (Old)")
    new_count, new_is_soft, new_counts = analyze_file(FILE_NEW, "新数据集 (New)")
    
    print(f"\n{'='*20} 综合对比结论 {'='*20}")
    
    # 1. 数据量对比
    if new_count > old_count:
        increase = new_count - old_count
        percent = (increase / old_count) * 100
        print(f"数据量提升: +{increase:,} 条 ({percent:.1f}%)")
        print("说明：回译增强或数据合并生效了。")
    elif new_count == old_count:
        print("数据量持平: 无新增样本。")
        print("说明：可能只做了软标签转换，没做增强；或者增强脚本未运行。")
    else:
        print(f"数据量减少: -{old_count - new_count:,} 条")
        print("说明：去重操作移除了大量重复数据 (这是好事！)。")

    # 2. 软标签对比
    if new_is_soft and not old_is_soft:
        print("标签升级: 硬标签 -> 软标签")
        print("说明：新模型能学习标注者的不确定性，理论上泛化性更强。")
    elif new_is_soft and old_is_soft:
        print("标签保持: 均为软标签")
    elif not new_is_soft:
        print("警告: 新数据似乎不是软标签格式！请检查转换脚本。")

    # 3. 少数类改善情况 (如果做了增强)
    if new_count > old_count and old_counts is not None and new_counts is not None:
        # 比较最少类的增长
        old_min = np.min(old_counts)
        new_min = np.min(new_counts)
        if new_min > old_min:
            print(f"少数类改善: 最少类的样本数从 {int(old_min)} 增加到 {int(new_min)}")
            print("说明：回译增强成功覆盖了长尾类别，有助于提升 Macro F1。")
        else:
            print(f"少数类未改善: 最少类的样本数从 {int(old_min)} 变为 {int(new_min)}")
            print("说明：回译增强可能未覆盖最少类，或增强效果有限。")

if __name__ == "__main__":
    main()
