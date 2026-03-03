import os
os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'


import json
from pathlib import Path
from transformers import AutoTokenizer
import torch
import numpy as np
from model import EmotionClassifier
from peft import LoraConfig, get_peft_model, TaskType

# 配置（与训练代码保持一致）
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN = 128
MODEL_PATH= 'best_model_lora.pth'  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LORA = True 

def _load_label_mapping():
    possible_paths = ['data/label_mapping.json', './label_mapping.json', '../data/label_mapping.json']
    for p in possible_paths:
        if Path(p).exists():
            with open(p, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            return [label_map[str(i)] for i in range(len(label_map))]

    # fallback: 28 labels
    return [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]


class PipelinePredictor:
    """轻量pipeline,加载tokenizer和模型,并提供 predict/get_results 接口"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_inited', False):
            return

        print("[Pipeline] 初始化预测器...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='C:/hf_cache')
        self.emotion_labels = _load_label_mapping()
        num_labels = len(self.emotion_labels)

        # 根据 USE_LORA 构建不同的模型架构
        if USE_LORA:
            print("-> 检测到 USE_LORA=True，构建 LoRA 架构...")
            
            # 加载基础模型 
            from transformers import AutoModelForSequenceClassification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                cache_dir='C:/hf_cache'
            )
            
            # 应用 LoRA 配置
            lora_config = LoraConfig(
                r=16,                
                lora_alpha=32,       
                target_modules=["query_proj", "value_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            
            # 包装模型
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
            
        else:
            print("   -> 使用全量模型架构...")
            # 使用你自定义的类
            self.model = EmotionClassifier(model_name=MODEL_NAME, num_labels=num_labels)

        # 加载权重
        model_path = MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件 {model_path} 未找到！当前目录：{os.listdir('.')}")
        
        print(f" -> 加载权重：{model_path} ...")
        state = torch.load(model_path, map_location='cpu')
        
        # 智能清理键名前缀
        clean_state = {}
        for k, v in state.items():
            name = k
            # 如果权重里已经有前缀，保持不变
            if name.startswith('base_model.model.'):
                pass 
            # 如果权重里没有前缀，且不是 'modules_to_save' 等特殊层，则加上前缀
            elif not name.startswith('modules_to_save'):
                name = 'base_model.model.' + name
            
            clean_state[name] = v
        
        print(f"已调整键名前缀：样本键名 -> {list(clean_state.keys())[0]}")
        
        # 加载权重 (strict=False 防止因微小差异报错，但会打印警告)
        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        if missing:
            print(f"未加载的键 (前5个): {missing[:5]}")
        if unexpected:
            print(f"多余的键 (前5个): {unexpected[:5]}")
        
        if not missing and not unexpected:
            print("权重完美加载！")
        elif not missing: 
            print("权重加载完成 (部分非关键键忽略)")

        # 移动到设备
        self.model = self.model.to(DEVICE)
        self.model.eval()
 
        print(f"模型已就绪 ({DEVICE}, FP32)")

        # 加载阈值
        self.thresholds = np.ones(num_labels) * 0.5
        project_root = Path(__file__).resolve().parent
        thresh_files = [
            project_root / 'results' / 'best_thresholds' / 'final_best_thresholds_verified.json',
            project_root / 'results' / 'best_thresholds' / 'best_thresholds.json'
        ]
        for tf in thresh_files:
            if tf.exists():
                with open(tf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'thresholds' in data:
                        self.thresholds = np.array(data['thresholds'])
                        print(f"阈值加载成功：{tf} -> [{self.thresholds.min():.2f}, {self.thresholds.max():.2f}]")
                        break
        
        self._inited = True
        print("[Pipeline] 初始化完成！")

    def predict_proba(self, text: str):
        text = text.strip()
        if not text:
            return np.zeros(len(self.emotion_labels))

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = enc['input_ids'].to(DEVICE)
        attention_mask = enc['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            if logits.dtype == torch.float16:
                logits = logits.float()
            
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        return probs

    def predict(self, text: str):
        probs = self.predict_proba(text)
        preds = (probs > self.thresholds).astype(int)
        return probs, preds

    def get_results(self, text: str, top_k: int = 10):
        probs, preds = self.predict(text)

        top_idx = np.argsort(probs)[-top_k:][::-1]
        top_results = {self.emotion_labels[i]: float(probs[i]) for i in top_idx}

        predicted = [self.emotion_labels[i] for i in range(len(probs)) if preds[i]]

        detailed = {self.emotion_labels[i]: {'probability': float(probs[i]), 'predicted': bool(preds[i])}
                    for i in range(len(probs))}

        return detailed, top_results, predicted


# module-level singleton
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PipelinePredictor()
    return _predictor


def predict(text: str, top_k: int = 10):
    p = get_predictor()
    return p.get_results(text, top_k)
