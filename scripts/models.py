from pydantic import BaseModel
from typing import List


class BulkStudyLink(BaseModel):
    study: str
    model_runs: List[str]
