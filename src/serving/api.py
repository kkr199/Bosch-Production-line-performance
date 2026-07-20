"""Authenticated FastAPI endpoint for the locally registered Bosch model."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.serving.model_service import InputContractError, ModelService


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description="All numeric model features required by /contract.")
    request_id: str | None = Field(default=None, max_length=100)


@lru_cache
def get_service() -> ModelService:
    return ModelService()


def require_api_key(x_api_key: str = Header(...)) -> None:
    expected_key = os.environ.get("BOSCH_API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


app = FastAPI(title="Bosch Failure Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/contract", dependencies=[Depends(require_api_key)])
def contract(service: ModelService = Depends(get_service)) -> dict[str, Any]:
    return service.contract()


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(
    request: PredictionRequest, service: ModelService = Depends(get_service)
) -> dict[str, Any]:
    try:
        result = service.predict(request.features, request.request_id)
    except InputContractError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return {
        "request_id": result.request_id,
        "model_version": result.model_version,
        "failure_probability": result.failure_probability,
        "predicted_failure": result.predicted_failure,
        "threshold": result.threshold,
        "latency_ms": result.latency_ms,
    }
