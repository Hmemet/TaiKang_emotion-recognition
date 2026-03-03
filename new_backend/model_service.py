import os
os.environ['HF_HOME'] = 'C:/hf_cache'  
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'
import torch
import json
import numpy as np
import os
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = 'microsoft/deberta-v3-base'

class EmotionPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_inited', False): return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Backend] 正在加载模型至设备: {self.device}")

       
        with open(BASE_DIR / "label_mapping.json", 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
            self.labels = [self.label_map[str(i)] for i in range(len(self.label_map))]
        
        with open(BASE_DIR / "models" / "best_thresholds.json", 'r', encoding='utf-8') as f:
            self.thresholds = np.array(json.load(f)['thresholds'])

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

       
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.1, bias="none",
            task_type=TaskType.SEQ_CLS
        )
        self.model = get_peft_model(base_model, lora_config)

      
        weights_path = BASE_DIR / "models" / "best_model_lora.pth"
        state = torch.load(weights_path, map_location='cpu')
        
        
        clean_state = {}
        for k, v in state.items():
            name = k if (k.startswith('base_model.model.') or k.startswith('modules_to_save')) else 'base_model.model.' + k
            clean_state[name] = v
        
        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(self.device).eval()
        
        self._inited = True
        print("[Backend] 模型初始化完成！")

    def predict(self, text: str, top_k: int = 5):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        
        preds = (probs > self.thresholds).astype(int)
        
        detected = []
        all_scores = []
        for i, (p, is_hit) in enumerate(zip(probs, preds)):
            item = {"label": self.labels[i], "score": round(float(p), 4)}
            all_scores.append(item)
            if is_hit:
                detected.append(item)

        if not detected:
            max_idx = np.argmax(probs)
            detected.append({"label": self.labels[max_idx], "score": round(float(probs[max_idx]), 4)})

        all_scores = sorted(all_scores, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return {
            "detected_emotions": detected,
            "top_k_scores": all_scores
        }