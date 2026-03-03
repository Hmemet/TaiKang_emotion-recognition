import os
import sys
import json
import time
from datetime import datetime


os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'utils' else script_dir
sys.path.insert(0, root_dir)

# 导入train函数
try:
    from train import train
    print("成功导入 train 函数")
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保 run_sampler_grid.py 和 train.py 在同一目录下，且 train.py 已正确修改。")
    sys.exit(1)

# 搜索配置
# 要测试的 Sampler Scale 列表
# 1.0 = 不缩放 (基准)
# 1.5, 2.0 = 增强少数类采样权重
SAMPLER_SCALES = [1.0, 1.5, 2.0]

# 要测试的 Small Class Boost 列表
# 1.2 = 轻度增强
# 1.5 = 中度增强 (默认)
# 2.0 = 重度增强
SMALL_CLASS_BOOSTS = [1.2, 1.5, 2.0]

# 短跑模式配置
SUBSET_SIZE = 3000   # 每次训练只用 3000 条数据
SHORT_EPOCHS = 1     # 只跑 1 个 Epoch

# 输出文件
RESULT_FILE = 'sampler_grid_results.json'

def run_single_config(scale, boost):
    """运行单次配置并返回 F1 分数"""
    print(f"\n{'='*50}")
    print(f"测试配置: Scale={scale}, Boost={boost}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # 调用 train 函数
        # 传入 sampler_scale 和 small_class_boost
        # 同时强制开启短跑模式 (use_subset=True, epochs_override=1)
        best_f1 = train(
            sampler_scale=scale,
            small_class_boost=boost,
            use_subset=True,
            subset_size=SUBSET_SIZE,
            epochs_override=SHORT_EPOCHS
            # focal_gamma 不传，使用 train.py 中的默认值
        )
        
        elapsed = time.time() - start_time
        
        print(f"完成 | Scale={scale}, Boost={boost} | Macro F1: {best_f1:.4f} | 耗时: {elapsed:.1f}s")
        
        return {
            "sampler_scale": scale,
            "small_class_boost": boost,
            "macro_f1": float(best_f1),
            "time_seconds": round(elapsed, 2),
            "status": "success"
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"失败 | Scale={scale}, Boost={boost} | 错误: {e}")
        return {
            "sampler_scale": scale,
            "small_class_boost": boost,
            "macro_f1": 0.0,
            "time_seconds": round(elapsed, 2),
            "status": "failed",
            "error": str(e)
        }

def main():
    print("开始 Sampler 超参数网格搜索 (短跑模式)")
    print(f"测试范围:")
    print(f"- Sampler Scales: {SAMPLER_SCALES}")
    print(f"- Small Class Boosts: {SMALL_CLASS_BOOSTS}")
    print(f"短跑配置: Data={SUBSET_SIZE}, Epochs={SHORT_EPOCHS}")
    print(f"结果将保存至: {RESULT_FILE}")
    
    results = []
    total_start = time.time()
    
    # 双重循环遍历所有组合
    for scale in SAMPLER_SCALES:
        for boost in SMALL_CLASS_BOOSTS:
            res = run_single_config(scale, boost)
            results.append(res)
            
            # 实时显示当前最佳
            valid_results = [r for r in results if r['status'] == 'success']
            if valid_results:
                best = max(valid_results, key=lambda x: x['macro_f1'])
                print(f"\n[当前最佳] Scale={best['sampler_scale']}, Boost={best['small_class_boost']} -> F1={best['macro_f1']:.4f}")

    total_elapsed = time.time() - total_start
    
    # 整理最终报告
    valid_results = [r for r in results if r['status'] == 'success']
    
    if not valid_results:
        print("\n所有测试均失败!请检查错误日志。")
        return

    best_result = max(valid_results, key=lambda x: x['macro_f1'])
    
    final_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "subset_size": SUBSET_SIZE,
            "epochs": SHORT_EPOCHS,
            "sampler_scales_tested": SAMPLER_SCALES,
            "boosts_tested": SMALL_CLASS_BOOSTS
        },
        "all_results": results,
        "best_result": best_result,
        "total_time_seconds": round(total_elapsed, 2)
    }
    
    # 保存结果
    out_path = os.path.join(root_dir, 'results', 'sampler_grid_results')
    os.makedirs(out_path, exist_ok=True)
    full_result_file = os.path.join(out_path, RESULT_FILE)
    with open(full_result_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "="*50)
    print("网格搜索完成!")
    print(f"总耗时: {total_elapsed:.1f} 秒")
    print(f"详细结果已保存至: {full_result_file}")
    print("="*50)
    print("推荐最佳配置:")
    print(f"   - SAMPLER_SCALE: {best_result['sampler_scale']}")
    print(f"   - SMALL_CLASS_BOOST: {best_result['small_class_boost']}")
    print(f"   - 预估 Macro F1: {best_result['macro_f1']:.4f}")
    print("="*50)
    print("    下一步操作:")
    print(f"   1. 打开 train.py")
    print(f"   2. 将默认配置修改为:")
    print(f"      SAMPLER_SCALE = {best_result['sampler_scale']}")
    print(f"      SMALL_CLASS_BOOST = {best_result['small_class_boost']}")
    print(f"   3. 运行 python train.py 进行全量训练！")

if __name__ == "__main__":
    main()