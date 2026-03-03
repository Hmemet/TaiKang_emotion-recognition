import os, sys, json, time
os.environ['HF_HOME'] = 'C:/hf_cache'
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'utils' else script_dir
sys.path.insert(0, root_dir)
from train import train

# 固定其他参数
FIXED_GAMMA = 0.0      
FIXED_SCALE = 1.0       
FIXED_BOOST = 1.0

# 搜索空间
LR_LIST = [1e-5, 2e-5, 3e-5]
DROPOUT_LIST = [0.1, 0.2, 0.3] 

SUBSET_SIZE = 4000 
EPOCHS = 2   

RESULTS_OUTPUT_ROOT = os.path.join('results')

def get_results_path(subfolder, filename):
    output_dir = os.path.join(RESULTS_OUTPUT_ROOT, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename) 

results = []
print("开始多数类增强网格搜索 (Gamma=0, 搜索 LR & Dropout)...")

for lr in LR_LIST:
    for drop in DROPOUT_LIST:
        print(f"\n测试: LR={lr}, Dropout={drop}")
        try:
            f1 = train(
                focal_gamma=FIXED_GAMMA,
                lr=lr,
                sampler_scale=FIXED_SCALE,
                small_class_boost=FIXED_BOOST,
                use_subset=True,
                subset_size=SUBSET_SIZE,
                epochs_override=EPOCHS,
                dropout_rate=drop,
            )
            results.append({"lr": lr, "dropout": drop, "f1": f1})
            best = max(results, key=lambda x: x['f1'])
            print(f"   F1={f1:.4f} | 当前最佳: LR={best['lr']}, Drop={best['dropout']} -> {best['f1']:.4f}")
        except Exception as e:
            print(f"   错误：{e}")

# 保存
out_path = get_results_path('majority_boost', 'majority_boost_results.json')
with open(out_path, 'w') as f:
    json.dump({"best": max(results, key=lambda x: x['f1']), "all": results}, f, indent=2)
print(f"\n结果已保存到 {out_path}")