"""Endpoint para cálculo de aderência."""
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
import pickle


MODEL_PATH = Path(__file__).resolve().parents[3] / "model.pkl"
TEST_DATASET_PATH = Path(__file__).resolve().parents[4] / "datasets" / "credit_01" / "test.gz"

router = APIRouter(prefix="/aderencia", tags=["aderencia"])


class AdherenceRequest(BaseModel):
    """Payload do endpoint de aderência."""

    dataset_path: str = Field(
        ...,
        description=(
            "Caminho local do dataset que será avaliado. Aceita caminhos absolutos ou relativos "
            "ao diretório atual do projeto e deve apontar para um arquivo em formato CSV, GZIP ou Parquet."
        ),
        json_schema_extra={"example": "datasets/credit_01/train.gz"},
    )


class AdherenceResponse(BaseModel):
    """Resposta do teste de aderência de scores."""

    ks_statistic: float = Field(
        ..., description="Estatística do teste KS, que mede a distância entre as distribuições de score."
    )
    p_value: float = Field(
        ...,
        description="Valor-p associado ao teste KS; valores baixos sugerem diferença estatisticamente relevante.",
    )
    n_registros_input: int = Field(..., description="Quantidade de registros presentes na base fornecida como entrada.")
    n_registros_teste: int = Field(..., description="Quantidade de registros da base de teste do modelo.")
    score_media_input: float = Field(..., description="Média dos scores gerados para a base de entrada.")
    score_media_teste: float = Field(..., description="Média dos scores gerados para a base de teste do modelo.")

    score_bins: List[float] = Field(..., description="Eixo X (probabilidades de 0 a 1) para plotagem.")
    density_input: List[float] = Field(..., description="Densidade da distribuição dos scores da base de entrada.")
    density_teste: List[float] = Field(..., description="Densidade da distribuição dos scores da base de teste.")
    cdf_input: List[float] = Field(..., description="Distribuição Acumulada (CDF) da base de entrada.")
    cdf_teste: List[float] = Field(..., description="Distribuição Acumulada (CDF) da base de teste.")

    categorias_desconhecidas_input: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Quantidade de valores substituídos por NaN, por coluna, na base de entrada, por "
            "conterem categorias não vistas durante o treino do modelo. Útil para auditar o quanto "
            "da métrica de aderência pode estar sendo influenciada por essa sanitização em vez de "
            "refletir 100% os dados originais."
        ),
    )
    categorias_desconhecidas_teste: Dict[str, int] = Field(
        default_factory=dict,
        description="Mesma contagem que `categorias_desconhecidas_input`, mas para a base de teste do modelo.",
    )
    total_substituicoes_input: int = Field(
        0, description="Soma de todas as substituições feitas na base de entrada (todas as colunas)."
    )
    total_substituicoes_teste: int = Field(
        0, description="Soma de todas as substituições feitas na base de teste (todas as colunas)."
    )


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the persisted model once and reuse it across requests."""

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def _resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve absolute or relative paths provided in the request."""

    candidate = Path(dataset_path).expanduser()
    candidates = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([
            (Path.cwd() / candidate).resolve(),
            (Path(__file__).resolve().parents[3] / candidate).resolve(),
            (Path(__file__).resolve().parents[4] / candidate).resolve(),
        ])

    for resolved in candidates:
        if resolved.exists():
            return resolved

    raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {dataset_path}")


def _read_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read a local CSV-like dataset with gzip support."""

    if dataset_path.suffix == ".gz":
        frame = pd.read_csv(dataset_path, compression="gzip")
    elif dataset_path.suffix in {".csv", ".txt"}:
        frame = pd.read_csv(dataset_path)
    elif dataset_path.suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    else:
        raise HTTPException(status_code=422, detail="Formato de arquivo não suportado.")

    return frame.replace({None: np.nan})


def _prepare_features(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Keep only the features expected by the model and align the column order."""

    if "TARGET" in frame.columns:
        frame = frame.drop(columns=["TARGET"])
    if "REF_DATE" in frame.columns:
        frame = frame.drop(columns=["REF_DATE"])

    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is not None:
        missing = [column for column in expected_columns if column not in frame.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Colunas ausentes para o modelo: {missing}")
        return frame.loc[:, list(expected_columns)]

    return frame


def _find_column_transformer(estimator: Any) -> "ColumnTransformer | None":
    """Recursively search a fitted pipeline/estimator for a ColumnTransformer step."""

    if isinstance(estimator, ColumnTransformer):
        return estimator

    candidates = []
    if hasattr(estimator, "steps"):  # sklearn Pipeline
        candidates.extend(step for _, step in estimator.steps)
    if hasattr(estimator, "named_steps"):
        candidates.extend(estimator.named_steps.values())
    if hasattr(estimator, "estimator") and estimator.estimator is not None:
        candidates.append(estimator.estimator)
    if hasattr(estimator, "estimators_"):
        candidates.extend(estimator.estimators_)

    for candidate in candidates:
        found = _find_column_transformer(candidate)
        if found is not None:
            return found
    return None


def _known_categories_by_column(
    column_transformer: "ColumnTransformer", reference_columns: List[str]
) -> Dict[str, set]:
    """Map each input column name to the categories the fitted encoder saw during training.

    Only covers transformers that expose `categories_` (OrdinalEncoder, OneHotEncoder, etc.),
    which is exactly what raises the "Found unknown categories" error we're guarding against.

    The `columns` selector stored on a fitted ColumnTransformer can be either a list of
    column *names* or a list of column *positions* (ints), depending on how the pipeline
    was built. `reference_columns` (the columns of the frame in the same order used at fit
    time, i.e. `model.feature_names_in_`) lets us resolve positional selectors back to the
    real column name — otherwise integer keys silently never match `frame.columns` (strings)
    and the corresponding column never gets sanitized.
    """

    known: Dict[str, set] = {}
    for _, transformer, columns in getattr(column_transformer, "transformers_", []):
        categories = getattr(transformer, "categories_", None)
        if categories is None:
            continue

        raw_columns = [columns] if isinstance(columns, (str, int, np.integer)) else list(columns)

        column_names: List[str] = []
        for col in raw_columns:
            if isinstance(col, (int, np.integer)) and 0 <= int(col) < len(reference_columns):
                column_names.append(reference_columns[int(col)])
            else:
                column_names.append(col)

        for column_name, cats in zip(column_names, categories):
            known[column_name] = set(cats)
    return known


def _sanitize_unknown_categories(frame: pd.DataFrame, model: Any) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Replace categorical values not seen during training with NaN before scoring.

    This avoids relying on parsing sklearn's error message (and its transformer-relative
    column index, which does not map back to `frame.columns` in a multi-transformer
    ColumnTransformer) to figure out which column needs fixing.

    Returns the sanitized frame plus a dict of {coluna: quantidade_substituida}, so the
    caller can report how much of the input was altered before scoring.
    """

    counts: Dict[str, int] = {}

    column_transformer = _find_column_transformer(model)
    if column_transformer is None:
        return frame, counts

    known_categories = _known_categories_by_column(column_transformer, list(frame.columns))
    if not known_categories:
        return frame, counts

    frame = frame.copy()
    for column_name, known_values in known_categories.items():
        if column_name not in frame.columns:
            continue
        is_unknown = frame[column_name].notna() & ~frame[column_name].isin(known_values)
        n_unknown = int(is_unknown.sum())
        if n_unknown:
            frame.loc[is_unknown, column_name] = np.nan
            counts[column_name] = counts.get(column_name, 0) + n_unknown
    return frame, counts


def _blank_unknown_value_wherever_found(frame: pd.DataFrame, unknown_value: str) -> Dict[str, int]:
    """Fallback: null out `unknown_value` in every object/category column that contains it.

    Used only when `_sanitize_unknown_categories` didn't already prevent the error (e.g. a
    column selector shape we don't recognize). Since we no longer trust the positional column
    index from sklearn's error message, we search by value instead of by index. Returns a dict
    of {coluna: quantidade_substituida} so the caller can add it to the running counter.
    """

    counts: Dict[str, int] = {}
    for column_name in frame.columns:
        if frame[column_name].dtype != object:
            continue
        mask = frame[column_name] == unknown_value
        n_matches = int(mask.sum())
        if n_matches:
            frame.loc[mask, column_name] = np.nan
            counts[column_name] = counts.get(column_name, 0) + n_matches
    return counts


def _score_dataset(frame: pd.DataFrame, model: Any) -> tuple[pd.Series, Dict[str, int]]:
    """Generate probability scores for a dataset using the model pipeline.

    Returns the scores plus a dict of {coluna: quantidade_substituida} covering both the
    upfront sanitization and any fallback substitutions triggered by retries.
    """

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="O modelo carregado não suporta predict_proba().")

    feature_frame = _prepare_features(frame, model)
    feature_frame, sanitize_counts = _sanitize_unknown_categories(feature_frame, model)

    max_retries = 5  # apenas uma rede de segurança para casos que a sanitização não cobriu
    last_error = None
    for _ in range(max_retries):
        try:
            scores = model.predict_proba(feature_frame)[:, 1]
            return pd.Series(scores, index=feature_frame.index), sanitize_counts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            match = re.search(r"Found unknown categories \[(.*?)\] in column", str(exc))
            if not match:
                raise HTTPException(status_code=500, detail=f"Falha ao escorar a base: {exc}") from exc

            unknown_values = [value.strip(" '\"") for value in match.group(1).split(",")]
            changed = False
            for value in unknown_values:
                fallback_counts = _blank_unknown_value_wherever_found(feature_frame, value)
                for column_name, n in fallback_counts.items():
                    sanitize_counts[column_name] = sanitize_counts.get(column_name, 0) + n
                    changed = True
            if not changed:
                raise HTTPException(status_code=500, detail=f"Falha ao escorar a base: {exc}") from exc

    raise HTTPException(
        status_code=500,
        detail=f"Muitas categorias desconhecidas mesmo após sanitização e fallback: {last_error}",
    )


@router.post(
    "/",
    response_model=AdherenceResponse,
    summary="Avalia a aderência da distribuição de scores",
    description=(
        "Compara a distribuição de scores de um dataset local com a base de teste do modelo usando o "
        "teste Kolmogorov-Smirnov (KS). A API calcula a distância entre as distribuições e retorna "
        "estatísticas de centralidade para apoiar a análise de aderência do score em produção."
    ),
    response_description="Resultado do teste de aderência e métricas comparativas de score entre a base de entrada e a base de teste.",
)
def calculate_aderencia(payload: AdherenceRequest = Body(...)) -> AdherenceResponse:
    """Compara a distribuição de scores de uma base local com a do conjunto de teste do modelo."""

    if not payload.dataset_path:
        raise HTTPException(status_code=422, detail="dataset_path é obrigatório.")

    model = _load_model()
    dataset_path = _resolve_dataset_path(payload.dataset_path)
    frame_input = _read_dataset(dataset_path)
    frame_test = _read_dataset(TEST_DATASET_PATH)

    if frame_input.empty or frame_test.empty:
        raise HTTPException(status_code=422, detail="Dataset vazio ou sem registros suficientes.")

    scores_input, unknown_counts_input = _score_dataset(frame_input, model)
    scores_test, unknown_counts_test = _score_dataset(frame_test, model)

    ks_statistic, p_value = ks_2samp(scores_input, scores_test)

    bins = np.linspace(0, 1, 101)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2

    hist_input, _ = np.histogram(scores_input, bins=bins, density=True)
    hist_test, _ = np.histogram(scores_test, bins=bins, density=True)

    cdf_input = np.cumsum(hist_input * np.diff(bins))
    cdf_test = np.cumsum(hist_test * np.diff(bins))

    return AdherenceResponse(
        ks_statistic=float(ks_statistic),
        p_value=float(p_value),
        n_registros_input=int(len(scores_input)),
        n_registros_teste=int(len(scores_test)),
        score_media_input=float(scores_input.mean()),
        score_media_teste=float(scores_test.mean()),
        score_bins=bin_midpoints.tolist(),
        density_input=hist_input.tolist(),
        density_teste=hist_test.tolist(),
        cdf_input=cdf_input.tolist(),
        cdf_teste=cdf_test.tolist(),
        categorias_desconhecidas_input=unknown_counts_input,
        categorias_desconhecidas_teste=unknown_counts_test,
        total_substituicoes_input=sum(unknown_counts_input.values()),
        total_substituicoes_teste=sum(unknown_counts_test.values()),
    )