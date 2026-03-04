import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class EmotionClassifier(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=28,
                 dropout_rate=0.2):
        super(EmotionClassifier, self).__init__()   
        #加载配置
        self.config = AutoConfig.from_pretrained(model_name)  
        #加载模型时开启 gradient_checkpointing
        self.deberta = AutoModel.from_pretrained(model_name)

        ##self.deberta.gradient_checkpointing_enable()

        #5-Dropout 集成
        print(f"[Model] 启用 5-Dropout 集成 (rate={dropout_rate})")
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(5)
        ])    
        # 单Dropout
        '''self.dropout = nn.Dropout(dropout_rate) 
        print(f"[Model] 启用单 Dropout (rate={dropout_rate})")'''
        self.classifier = nn.Linear(self.config.hidden_size, num_labels )       
        # 初始化权重 
        self._init_weights(self.classifier)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, input_ids, attention_mask):
        # 当开启 gradient_checkpointing 时，forward 依然正常工作
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)       
        # 获取 [CLS] token 的特征
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # 类型转换 (防止混合精度训练时的 dtype 不匹配)
        pooled_output = pooled_output.to(self.classifier.weight.dtype)
        # 5-Dropout 集成预测
        logits = sum([self.classifier(dropout(pooled_output)) for dropout in self.dropouts]) / 5
        # 标准流程：Dropout -> Linear
        '''x = self.dropout(pooled_output)
        logits = self.classifier(x)'''
        return logits