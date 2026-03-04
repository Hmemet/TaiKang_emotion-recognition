# 注意:
import os
os.environ['HF_HOME'] = 'C:/hf_cache'  # 使用英文路径
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'
这是我用户名是中文采取的应急方案
用户名英文的可以去掉

# 1.train.py
## 核心配置区进行大部分参数条件
## 少部分如Sampler 参数 需进train内（dropout的default值在函数头）
（183）SAMPLER_SCALE = sampler_scale if sampler_scale is not None else float(os.environ.get('SAMPLER_SCALE', '1.2'))
SMALL_CLASS_BOOST = small_class_boost if small_class_boost is not None else float(os.environ.get('SMALL_CLASS_BOOST', '1.2'))

（34）  USE_LORA = False  #设为 True 开启 LoRA，False 关闭       lora模式开关
（255）    RESUME_FROM_CHECKPOINT = False      续训模式开关

## 不同模式产物及测试区别
lora 模式：final_report_lora.csv  evaluate_lora.py  evaluation_visuals_lora.png  best_model_lora.pth
全参(关闭lora)模式:final_report.csv  evaluate.py  evaluation_visuals.png  best_model.pth

# 2.utils
## back_translate_aug.py
回译脚本
## grid_search_gamma.py
Gamma网格短跑
## majority_boost_grid.py
lr,dropout网格短跑   //没完成，无法跑
## re_eval.py
阈值方法对比测试
## run_sampler_grid.py
sampler_scales,small_class_boosts网格短跑  
## check
### verify.py
标签打乱测试,检测是否存在数据泄露
### test_consistency.py
### test_on_holdout.py
### test_robustness.py


# 3.data数据集改动
## /processed
### count.py 
数数据行数
### compare_datasets.py
新老训练集测试对比
### train_soft.csv
（未合并）软标签化输出
### train_soft_corrected.csv
合并后软标签化输出
### train_soft_augmented.csv
软标签化输出回译（增加50%）

## /raw/soft_labels
### convert_parquet.py
软标签化输出
### fix_parquet_aggregate.py
合并后软标签化输出
### goemotions.parquet
https://huggingface.co/datasets/google-research-datasets/go_emotions/blob/main/raw/train-00000-of-00001.parquet

# 4.copy
copy中有best_model.pth和best_model_lora.pth的最优备份，和几次train代码大改前的备份

# 5.gradio
有pipeline.py，通过gradio_app.py可直接生成可视化的 Web 应用

# 6.results与picures
## result
网格结果，report和阈值等。其中best_thresholds.json是训练输出，会因滞后性等原因不匹配。
而final_best_thresholds_verified.json是evaluate_lora.py输出，准。