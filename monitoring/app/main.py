"""Main module."""
import uvicorn
from fastapi import FastAPI
from .api.routers import router


app = FastAPI(
    title="Monitoramento de modelos de crédito",
    version="1.0.0",
    description=(
        "API para monitorar a qualidade operacional e estatística de um modelo de score de crédito. "
        "A solução oferece dois fluxos principais: (1) avaliação da performance em lote com AUC-ROC e "
        "volumetria mensal; e (2) validação da aderência da distribuição de scores em relação à base de teste."
    ),
)


@app.get(
    "/",
    summary="Status da API",
    description="Endpoint de verificação da disponibilidade da API e do ambiente de monitoramento.",
)
def read_root():
    """Retorna uma mensagem simples indicando que a API está ativa."""
    return {"status": "online", "message": "API de monitoramento de modelos ativa"}


app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
