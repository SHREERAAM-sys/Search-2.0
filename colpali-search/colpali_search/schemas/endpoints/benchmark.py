from pydantic import BaseModel


class BenchmarkResponse(BaseModel):
    average_recall_score: float
    precision_score: float
    mrr_score: float
