import datetime

from colpali_search.custom_types import CustomBaseModel


class ConfigureIndexResponse(CustomBaseModel):
    id: int
    strategy_name: str
    created_at: datetime.datetime


class ResetIndexResponse(CustomBaseModel):
    status: str
    message: str
    reset_time: datetime.datetime


class GetCurrentIndexStrategyResponse(CustomBaseModel):
    status: str
    name: str
    message: str
