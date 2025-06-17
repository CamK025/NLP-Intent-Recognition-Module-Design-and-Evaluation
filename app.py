"""
Start server:
    uvicorn app:app --host 127.0.0.1 --port 8080
"""

import uvicorn, numpy as np, tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional                   
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ---------- model & tokenizer ----------
MODEL_DIR = "./intent_model_tf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ---------- request / response ----------
class PredictRequest(BaseModel):
    text: str = Field(..., description="User input to classify")

class PredictResponse(BaseModel):
    intentId:   int   = Field(..., description="Predicted intent numeric ID")
    confidence: float = Field(..., ge=0, le=1)

class Status(BaseModel):
    code:    int
    message: str
    details: Optional[List] = None       

# ---------- FastAPI app ----------
app = FastAPI(
    title="Intent Classification API",
    version="0.1",
    docs_url="/",                 # Swagger UI root
    openapi_url="/openapi.json"   # OpenAPI spec
)

@app.post(
    "/v1/intents:predict",
    response_model      = PredictResponse,
    responses           = {
        400: {"model": Status, "description": "Invalid argument"},
        500: {"model": Status, "description": "Internal error"},
    },
)
def predict(req: PredictRequest):
    # Basic validation
    if not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": 3, "message": "text field is empty", "details": []},
        )

    try:
        # Tokenize & predict
        inputs = tokenizer(
            req.text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="tf",
        )
        logits = model(inputs, training=False).logits
        probs  = tf.nn.softmax(logits, axis=-1).numpy()[0]

        return PredictResponse(
            intentId=int(np.argmax(probs)),
            confidence=float(np.max(probs)),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": 13, "message": str(e), "details": []},
        ) from e


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080)

