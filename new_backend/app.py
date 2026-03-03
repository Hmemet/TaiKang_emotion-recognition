import time
import logging
import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
os.environ.pop("TRANSFORMERS_CACHE", None)

from model_service import EmotionPredictor
from utils import clean_text, logger

app = FastAPI(
    title="细粒度情绪识别系统 API",
    description="基于 DeBERTa-v3 LoRA 微调的 28 类情绪识别后端服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    predictor = EmotionPredictor()
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    
    predictor = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, json_schema_extra={"example": "我今天真的很开心！"})
    top_k: Optional[int] = Field(5, description="返回前 K 个最高分的情绪")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="待预测文本列表")
    top_k: Optional[int] = Field(5, description="每条文本返回前 K 个最高分的情绪")

class EmotionResult(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    status: str
    text_processed: str
    detected_emotions: List[EmotionResult]
    top_k_scores: List[EmotionResult]
    cost_ms: float


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未就绪，请检查服务端日志")

    start_time = time.time()
    
    try:
       
        raw_text = request.text
        processed_text = clean_text(raw_text)
        
        if not processed_text:
            raise HTTPException(status_code=400, detail="清洗后的文本内容为空")

       
        inference_result = predictor.predict(processed_text, top_k=request.top_k)
      
        duration = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "text_processed": processed_text,
            "detected_emotions": inference_result["detected_emotions"],
            "top_k_scores": inference_result["top_k_scores"],
            "cost_ms": duration
        }

    except Exception as e:
        logger.error(f"推理请求处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

@app.get("/api/health")
async def health_check():
    """健康检查接口，用于 Docker 监控"""
    return {
        "status": "healthy" if predictor else "unhealthy",
        "device": str(predictor.device) if predictor else "unknown",
        "timestamp": time.time()
    }


@app.post("/api/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未就绪，请检查服务端日志")

    start_time = time.time()
    results = []

    try:
        for index, raw_text in enumerate(request.texts):
            processed_text = clean_text(raw_text)
            if not processed_text:
                results.append({
                    "index": index,
                    "status": "error",
                    "message": "清洗后的文本内容为空",
                    "text_processed": "",
                    "detected_emotions": [],
                    "top_k_scores": []
                })
                continue

            inference_result = predictor.predict(processed_text, top_k=request.top_k)
            results.append({
                "index": index,
                "status": "success",
                "message": "ok",
                "text_processed": processed_text,
                "detected_emotions": inference_result["detected_emotions"],
                "top_k_scores": inference_result["top_k_scores"]
            })

        return {
            "status": "success",
            "count": len(results),
            "results": results,
            "cost_ms": round((time.time() - start_time) * 1000, 2)
        }

    except Exception as e:
        logger.error(f"批量推理请求处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)