"""Endpoint para cálculo de performance."""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple # ADICIONADO: Tuple
import pickle

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score, roc_curve # ADICIONADO: roc_curve



MODEL_PATH = Path(__file__).resolve().parents[3] / "model.pkl"

router = APIRouter(prefix="/performance", tags=["performance"])


class PerformanceResponse(BaseModel):
    """Resposta da avaliação de performance do lote."""

    volumetria: Dict[str, int] = Field(
        ..., description="Quantidade de registros por mês presente na base de monitoramento, em ordem cronológica."
    )
    auc_roc: float = Field(..., description="Área sob a curva ROC calculada para o lote informado.")
    
    # ADICIONADO: Novas listas para permitir o desenho da curva ROC no front-end
    fpr: List[float] = Field(..., description="Lista com a Taxa de Falsos Positivos (FPR) nos diferentes limiares.")
    tpr: List[float] = Field(..., description="Lista com a Taxa de Verdadeiros Positivos (TPR) nos diferentes limiares.")
    
    total_registros: int = Field(..., description="Número total de registros avaliados no lote.")


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the persisted model once and reuse it across requests."""

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def _prepare_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a model-ready DataFrame from the incoming batch."""

    frame = pd.DataFrame.from_records(records)
    frame = frame.replace({None: np.nan})
    return frame


def _build_monthly_volume(frame: pd.DataFrame) -> Dict[str, int]:
    """Count records by REF_DATE month, keeping chronological order."""

    if "REF_DATE" not in frame.columns:
        raise HTTPException(status_code=422, detail="Campo REF_DATE é obrigatório.")

    ref_dates = pd.to_datetime(frame["REF_DATE"], errors="coerce", utc=True)
    if ref_dates.isna().any():
        raise HTTPException(status_code=422, detail="Há valores inválidos em REF_DATE.")

    month_counts = (
        ref_dates.dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
        .value_counts(sort=False)
        .sort_index()
    )
    return {month: int(count) for month, count in month_counts.items()}


# MODIFICADO: A função agora retorna a AUC e as duas listas (FPR e TPR)
def _compute_auc(frame: pd.DataFrame, model: Any) -> Tuple[float, List[float], List[float]]:
    """Score the batch and compute ROC AUC and Curve coordinates against TARGET."""

    if "TARGET" not in frame.columns:
        raise HTTPException(status_code=422, detail="Campo TARGET é obrigatório.")

    y_true = frame["TARGET"]
    if y_true.isna().any():
        raise HTTPException(status_code=422, detail="Há valores nulos em TARGET.")

    unique_targets = pd.Series(y_true).dropna().unique()
    if len(unique_targets) < 2:
        raise HTTPException(
            status_code=422,
            detail="É necessário haver as duas classes do alvo para calcular a AUC.",
        )

    feature_frame = frame.drop(columns=[column for column in ["REF_DATE", "TARGET"] if column in frame.columns])

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="O modelo carregado não suporta predict_proba().")

    try:
        scores = model.predict_proba(feature_frame)[:, 1]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Falha ao escorar a base: {exc}") from exc

    # Calculando a AUC e os pontos da Curva
    auc_value = float(roc_auc_score(y_true, scores))
    fpr_array, tpr_array, _ = roc_curve(y_true, scores)

    # Convertendo numpy arrays para listas Python puras (necessário para o JSON/FastAPI)
    return auc_value, fpr_array.tolist(), tpr_array.tolist()


@router.post(
    "/",
    response_model=PerformanceResponse,
    summary="Avalia a performance do modelo em um lote",
    description=(
        "Recebe um lote de registros do período monitorado, valida os campos críticos do payload e calcula "
        "a volumetria mensal junto com a AUC-ROC do modelo e os pontos da curva. Esse endpoint é usado para acompanhar a "
        "performance do score em um batch recente e detectar quedas de qualidade em produção."
    ),
    response_description="Volumetria mensal, métrica de performance e coordenadas da curva ROC do lote processado.",
)
def calculate_performance(
    records: List[Dict[str, Any]] = Body(...)
) -> PerformanceResponse:
    """Calcula volumetria mensal e AUC ROC para um lote de registros de monitoramento."""

    if not records:
        raise HTTPException(status_code=422, detail="A lista de registros não pode ser vazia.")

    frame = _prepare_frame(records)
    model = _load_model()

    volumetria = _build_monthly_volume(frame)
    
    # MODIFICADO: Desempacotando os 3 valores retornados pela função
    auc_roc, fpr, tpr = _compute_auc(frame, model)

    return PerformanceResponse(
        volumetria=volumetria,
        auc_roc=auc_roc,
        fpr=fpr,   # ADICIONADO
        tpr=tpr,   # ADICIONADO
        total_registros=len(frame),
    )