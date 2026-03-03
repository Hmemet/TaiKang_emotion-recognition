import os
import pandas as pd
from tqdm import tqdm

# 配置
INPUT_FILE = 'data/raw/soft_labels/goemotions.parquet'
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_soft.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义标准的 28 个情感类别名称
EMOTION_COLUMNS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 '{INPUT_FILE}'")
        return

    print(f"📂 发现文件: {INPUT_FILE}")
    print("⚙️ 正在读取 Parquet 文件...")

    try:
        df = pd.read_parquet(INPUT_FILE)
        print(f"✅ 读取成功！共 {len(df)} 条数据。")
        
        # 检查必要的列
        if 'text' not in df.columns:
            print("❌ 错误：缺少 'text' 列。")
            return
            
        # 检查情感列是否存在 (容错处理)
        missing_cols = [col for col in EMOTION_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"❌ 错误：缺少以下情感列: {missing_cols}")
            print(f"   实际存在的列: {df.columns.tolist()}")
            return
        
        print(f"✅ 检测到 {len(EMOTION_COLUMNS)} 个情感列，开始转换软标签...")

        processed_data = []
        num_classes = len(EMOTION_COLUMNS)
        
        # 预提取情感列的数据，加速循环
        emotion_data = df[EMOTION_COLUMNS].values
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
            text = row['text']
            
            # 获取当前行的情感向量 (0/1 列表)
            labels_vector = emotion_data[idx]
            
            # 找出所有为 1 的索引
            label_ids = [i for i, val in enumerate(labels_vector) if val == 1]
            
            if not label_ids:
                continue
            
            # 构建软标签向量 (归一化)
            # 逻辑：如果有 k 个标签，每个标签概率为 1/k
            soft_target = [0.0] * num_classes
            weight = 1.0 / len(label_ids)
            for lid in label_ids:
                soft_target[lid] = weight
            
            processed_data.append({
                'text': text,
                'labels': str(soft_target)
            })
            
        # 保存为 CSV
        out_df = pd.DataFrame(processed_data)
        out_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n🎉 转换成功！")
        print(f"📂 输出文件: {OUTPUT_FILE}")
        print(f"📊 有效样本数: {len(out_df)}")
        print(f"🔍 示例软标签: {out_df.iloc[0]['labels']}")
        print("\n✅ 下一步：继续运行回译增强或训练脚本！")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()