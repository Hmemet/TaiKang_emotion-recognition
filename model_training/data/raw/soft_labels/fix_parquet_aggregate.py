import os
import pandas as pd
from tqdm import tqdm

# 配置
INPUT_FILE = file_path = 'data/raw/soft_labels/goemotions.parquet'   # 那个 21 万行的文件
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_soft_corrected.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义 28 个情感列 (根据你的 parquet 列名调整)
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
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"📂 加载原始数据: {INPUT_FILE} ({os.path.getsize(INPUT_FILE)/1024/1024:.1f} MB)")
    df_raw = pd.read_parquet(INPUT_FILE)
    print(f"✅ 原始行数: {len(df_raw):,} (这是包含所有标注者的展开表)")

    print("⚙️ 正在按文本合并标注 (Aggregate)...")
    
    # 核心逻辑：按 text 分组，对情感列求均值 (即计算投票比例)
    # 这样 3 个人标 joy (1, 1, 0) -> 合并后 joy=0.67
    grouped = df_raw.groupby('text')[EMOTION_COLUMNS].mean().reset_index()
    
    print(f"✅ 合并后行数: {len(grouped):,} (这才是真正的评论数，应约为 5.8 万)")

    # 转换为软标签列表字符串
    processed_data = []
    for idx, row in tqdm(grouped.iterrows(), total=len(grouped), desc="格式化"):
        text = row['text']
        # 获取 28 个情感的概率值 (0.0 - 1.0)
        probs = row[EMOTION_COLUMNS].values.tolist()
        
        # 可选：过滤掉全 0 的样本 (如果没有情感标签)
        if sum(probs) == 0:
            continue
            
        processed_data.append({
            'text': text,
            'labels': str(probs) # 存储为 "[0.5, 0.0, ...]" 字符串
        })

    out_df = pd.DataFrame(processed_data)
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n🎉 成功！")
    print(f"📂 输出文件: {OUTPUT_FILE}")
    print(f"📊 最终有效样本数: {len(out_df):,}")
    print(f"🔍 示例软标签: {out_df.iloc[0]['labels']}")
    print("\n💡 下一步：修改 train.py 指向这个新文件，数据量将回归正常的 5.8 万！")

if __name__ == "__main__":
    main()