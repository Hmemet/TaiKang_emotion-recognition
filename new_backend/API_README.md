# 细粒度对话情绪识别服务 - API 使用说明

## 1. 项目概述
本项目提供一个基于 RESTful 架构的 API 服务，旨在从对话文本中识别细粒度的情绪类别（28 类）。后端采用 FastAPI 框架，模型层集成 DeBERTa-v3。

## 2. 基础信息
- **接口根地址**: `http://localhost:8000`
- **交互式文档**: `http://localhost:8000/docs` (Swagger UI)
- **数据格式**: JSON
- **字符编码**: UTF-8

## 3. 核心接口详情

### 3.1 情绪识别预测
**请求信息：**
- **URL**: `/predict`
- **方法**: `POST`
- **内容类型**: `application/json`

**请求参数示例：**
```json
{
  "text": "今天的天气真好，心情也很舒畅。"
}
