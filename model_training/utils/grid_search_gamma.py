import os
import sys
import json
import time
from datetime import datetime

# 设置环境变量
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'utils' else script_dir
sys.path.insert(0, root_dir)

from train import train

RESULTS_OUTPUT_ROOT = os.path.join('results')

def get_results_path(subfolder, filename):
    output_dir = os.path.join(RESULTS_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

# 要测试的 Gamma 值列表
# 0.0 = 加权 BCE 
# 0.5 = 轻度 Focal
# 1.0 = 中度 Focal
# 1.5 = 重度 Focal 
# 2.0 = 极度 Focal 
GAMMA_LIST = [0.8, 0.9, 1.0, 1.1, 1.2]

# 短跑配置
SUBSET_SIZE = 3000      # 每次只用 3000 条数据训练 
SHORT_EPOCHS = 1        # 只跑 1 个 Epoch
VAL_CSV = 'data/processed/val.csv' # 验证集依然用完整的，保证评估准确



def run_single_gamma(gamma):
    print(f"\n{'='*40}")
    print(f"开始测试 Gamma = {gamma}")
    print(f"{'='*40}")
    
    start_time = time.time()
    
    try:
       
        best_f1 = train(
            focal_gamma=gamma,
            epochs_override=SHORT_EPOCHS,
            use_subset=True,
            subset_size=SUBSET_SIZE
        )
        
        elapsed = time.time() - start_time
        
        print(f"Gamma={gamma} 测试完成 | Macro F1: {best_f1:.4f} | 耗时: {elapsed:.1f}s")
        
        return {
            "gamma": gamma,
            "macro_f1": float(best_f1),
            "time_seconds": elapsed
        }
        
    except Exception as e:
        print(f"Gamma={gamma} 测试失败: {e}")
        return {
            "gamma": gamma,
            "macro_f1": 0.0,
            "error": str(e)
        }

def main():
    print("开始 Focal Loss Gamma 网格搜索 (短跑模式)")
    print(f"测试列表: {GAMMA_LIST}")
    print(f"数据量: {SUBSET_SIZE} 条 | Epochs: {SHORT_EPOCHS}")
    
    results = []
    
    for gamma in GAMMA_LIST:
        res = run_single_gamma(gamma)
        results.append(res)
        
        # 简单排序显示当前最佳
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = max(valid_results, key=lambda x: x['macro_f1'])
            print(f"当前最佳: Gamma={best['gamma']} (F1={best['macro_f1']:.4f})")

    # 保存结果
    final_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "subset_size": SUBSET_SIZE,
            "epochs": SHORT_EPOCHS
        },
        "results": results,
        "best_gamma": max([r for r in results if 'error' not in r], key=lambda x: x['macro_f1'])['gamma'] if any('error' not in r for r in results) else None
    }
    
    out_path=get_results_path('gamma_grid_results','gamma_grid_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("网格搜索完成!")
    print(f"结果已保存至: {out_path}")
    
    if final_report['best_gamma'] is not None:
        print(f"推荐最佳 Gamma: {final_report['best_gamma']}")
        print(f"下一步：请使用此 Gamma 值修改 train.py 并进行全量训练！")

if __name__ == "__main__":
    main()