# gradio_app_final_fixed.py
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer
from model import EmotionClassifier
from functools import lru_cache
import time
from pathlib import Path
import hashlib
import os


os.environ['HF_HOME'] = 'C:/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*50)
print("情感分析系统配置")
print("="*50)
print(f"设备: {DEVICE}")
print(f"HF缓存: C:/hf_cache")
print(f"模型: {MODEL_NAME}")
print(f"最大长度: {MAX_LEN}")
print("="*50)


class OptimizedEmotionPredictor:
    """完全匹配训练配置的预测器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("初始化优化预测器...")
        
        # 先加载tokenizer
        self.load_tokenizer()
        
        # 加载标签映射
        self.load_label_mapping()
        
        # 加载模型
        self.load_model()
        
        # 阈值
        self.thresholds = np.ones(28) * 0.5
        
        self._initialized = True
        print("优化预测器初始化完成！")
    
    def load_tokenizer(self):
        """加载tokenizer - 完全匹配训练代码"""
        try:
            print("加载tokenizer...")
            
            # 使用和训练代码完全相同的方式加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir='C:/hf_cache'  # 指定缓存目录
            )
            
            print(f"Tokenizer加载成功")
            print(f" - 词汇表大小: {len(self.tokenizer)}")
            print(f" - 最大长度: {self.tokenizer.model_max_length}")
            
        except Exception as e:
            print(f"加载tokenizer失败: {e}")
           
            print("Tokenizer从镜像加载成功")
    
    def load_label_mapping(self):
        """加载情绪标签映射"""
        try:
            # 查找label_mapping.json
            possible_paths = [
                'data/label_mapping.json',
                './label_mapping.json',
                '../data/label_mapping.json'
            ]
            
            label_map = None
            for path in possible_paths:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        label_map = json.load(f)
                    print(f"从 {path} 加载标签映射")
                    break
            
            if label_map:
                self.emotion_labels = [label_map[str(i)] for i in range(len(label_map))]
            else:
                # 训练代码中用的28个标签
                self.emotion_labels = [
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
                    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                    'gratitude', 'grief', 'joy', 'love', 'nervousness',
                    'optimism', 'pride', 'realization', 'relief', 'remorse',
                    'sadness', 'surprise', 'neutral'
                ]
                print(" 使用默认情绪标签")
            
            print(f"加载了 {len(self.emotion_labels)} 个情绪标签")
            
        except Exception as e:
            print(f"加载标签映射失败: {e}")
            self.emotion_labels = [f"emotion_{i}" for i in range(28)]
    
    def load_model(self):
        """加载训练好的模型权重"""
        try:
            print("创建模型实例...")
            self.model = EmotionClassifier(model_name=MODEL_NAME, num_labels=28)
            
            # 加载训练好的权重
            model_path = 'best_model_lora.pth'
            if Path(model_path).exists():
                print(f"加载训练好的权重: {model_path}")
                # 先加载到CPU再移动到设备
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print("模型权重加载成功")
                
                # 显示模型信息
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"- 总参数: {total_params/1e6:.2f}M")
                print(f"- 可训练参数: {trainable_params/1e6:.2f}M")
                
            else:
                print(f"错误：找不到模型文件 {model_path}")
                print("请确保 best_model_lora.pth 在当前目录")
                print("当前目录内容:", os.listdir('.'))
                raise FileNotFoundError(f"模型文件 {model_path} 不存在")
            
            # 移动到设备
            self.model = self.model.to(DEVICE)
            self.model.eval()
            
            # GPU优化
            if DEVICE.type == 'cuda':
                self.model = self.model.half()
                print("启用半精度优化")
                
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise e
    
    @lru_cache(maxsize=128)
    def predict_proba(self, text):
        """预测概率 - 完全匹配训练代码的数据处理"""
        text = text.strip()
        if not text:
            return np.zeros(28)
        
        try:
            # 使用和训练代码完全相同的编码方式
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 移动到设备
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            # 推理
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                if logits.dtype == torch.float16:
                    logits = logits.float()
                probabilities = torch.sigmoid(logits)
            
            return probabilities[0].cpu().numpy()
            
        except Exception as e:
            print(f"预测失败: {e}")
            return np.zeros(28)
    
    def predict(self, text):
        """预测（使用阈值）"""
        probs = self.predict_proba(text)
        preds = (probs > self.thresholds).astype(int)
        return probs, preds
    
    def get_results(self, text, top_k=10):
        """获取格式化的结果"""
        probs, preds = self.predict(text)
        
        # 获取top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_results = {self.emotion_labels[i]: float(probs[i]) for i in top_indices}
        
        # 获取预测的情绪
        predicted = [self.emotion_labels[i] for i in range(len(probs)) if preds[i]]
        
        # 详细结果
        results = {}
        for i, label in enumerate(self.emotion_labels):
            results[label] = {
                'probability': float(probs[i]),
                'predicted': bool(preds[i])
            }
        
        return results, top_results, predicted

def create_beautiful_interface():
    """创建美观的Gradio界面"""
    
    # 优先使用 pipeline.py 中的统一预测接口
    try:
        from pipeline import get_predictor
        predictor = get_predictor()
        print("已使用 pipeline.get_predictor() 作为预测器")
    except Exception:
        try:
            predictor = OptimizedEmotionPredictor()
            print("使用内置 OptimizedEmotionPredictor 作为后备预测器")
        except Exception as e:
            print(f"初始化失败: {e}")
            raise e
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: linear-gradient(135deg, #0066cc 0%, #0099ff 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    """
    
    with gr.Blocks(css=custom_css, title="情感分析系统", theme=gr.themes.Soft()) as demo:
        
        # 头部
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
            # 多标签情感分析系统
            ### 基于 DeBERTa-v3-base | 28种情绪分类
            """)
        
        # 统计信息
        with gr.Row():
            with gr.Column(scale=1, elem_classes="stats-card"):
                gr.Markdown(f"### 设备\n**{DEVICE}**")
            with gr.Column(scale=1, elem_classes="stats-card"):
                gr.Markdown(f"### 情绪类别\n**{len(predictor.emotion_labels)}种**")
            with gr.Column(scale=1, elem_classes="stats-card"):
                gr.Markdown(f"### 最大长度\n**{MAX_LEN} tokens**")
        
        # 主界面
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    lines=4,
                    placeholder="在这里输入文本...",
                    label="输入文本",
                    elem_id="input-text"
                )
                
                with gr.Row():
                    predict_btn = gr.Button("预测情绪", variant="primary", size="lg")
                    clear_btn = gr.Button("清除", size="lg")
                
                top_k = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="显示Top-K情绪"
                )
                
                # 示例文本
                gr.Examples(
                    examples=[
                        ["I'm so excited about the new project! It's going to be amazing!"],
                        ["This is really frustrating, I can't believe this happened."],
                        ["I love spending time with my family, they're the best."],
                        ["That's interesting, tell me more about it."],
                        ["I feel neutral about this, nothing special."]
                    ],
                    inputs=text_input,
                    label="试试这些示例"
                )
            
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=5, label="检测到的情绪")
        
        with gr.Row():
            # 使用 plotly 图表代替 BarPlot，以提高兼容性和可控性
            bar_plot = gr.Plot(label="情绪概率分布")
        
        with gr.Row():
            json_output = gr.JSON(label="详细概率")
        
        # 预测函数
        def predict_emotion(text, k):
            try:
                print(f"predict_emotion called. text_len={len(text) if text else 0}, top_k={k}")
                if not text or not text.strip():
                    return None, {}, {}

                results, top_results, predicted = predictor.get_results(text, k)

                # 使用 plotly 生成图表
                if top_results:
                    emotions = list(top_results.keys())
                    probs = list(top_results.values())
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(x=probs[::-1], y=emotions[::-1], orientation='h'))
                    fig.update_layout(xaxis_title='概率', yaxis_title='情绪类型', height=400)
                else:
                    fig = None

                # 准备标签数据（返回前5个预测）
                label_dict = {}
                for emotion in predicted[:5]:
                    if emotion in results:
                        label_dict[emotion] = results[emotion]['probability']

                print(f"predict_emotion finished. top_results_count={len(top_results)} predicted_count={len(predicted)}")
                return fig, label_dict, results

            except Exception as e:
                print(f"predict_emotion 异常: {e}")
                import traceback; traceback.print_exc()
                return None, {}, {}
        
        # 清除函数 - 返回4个值
        def clear_all():
            try:
                print("clear_all called")
                return "", None, {}, {}
            except Exception as e:
                print(f"clear_all 异常: {e}")
                return "", None, {}, {}
        
        # 绑定事件
        predict_btn.click(
            fn=predict_emotion,
            inputs=[text_input, top_k],
            outputs=[bar_plot, label_output, json_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[text_input, bar_plot, label_output, json_output]
        )
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 支持的28种情绪
        admiration, amusement, anger, annoyance, approval, caring, confusion, 
        curiosity, desire, disappointment, disapproval, disgust, embarrassment, 
        excitement, fear, gratitude, grief, joy, love, nervousness, optimism, 
        pride, realization, relief, remorse, sadness, surprise, neutral
        """)
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*50)
    print("启动情感分析系统...")
    print("="*50)
    
    try:
        # 创建并启动应用
        demo = create_beautiful_interface()
        
        print("\n系统启动成功!")
        print("访问地址: http://127.0.0.1:7860")
        print("="*50 + "\n")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n解决方案:")
        print("1. 确保 best_model_lora.pth 在当前目录")
        print("2. 运行 train.py 确保模型正确训练")
        print("3. 检查网络连接")