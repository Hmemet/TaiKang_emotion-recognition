import pandas as pd
file_path = 'data/raw/soft_labels/goemotions.parquet' 

try:
    df = pd.read_parquet(file_path)
    print(f"📊 goemotions.parquet 真实行数: {len(df):,}")
    print(f"   列名: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"❌ 找不到文件: {file_path}")
    print("💡 提示：请确认 parquet 文件的具体位置")

df_old = pd.read_csv('data/processed/train_augmented.csv')
df_new = pd.read_csv('data/processed/train_soft_corrected.csv')
print(f"老文件行数: {len(df_old)}")
print(f"新文件行数: {len(df_new)}")