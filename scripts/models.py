from pydantic import BaseModel
from typing import List


class BulkStudyLink(BaseModel):
    study: str
    model_runs: List[str]


class PermanentlyDelete(BaseModel):
    # The ID of the dataset to delete files from permanently
    dataset_id: str
    # A list of regexes relative to the dataset base path e.g. ["name.*", "scenario[0-9]*/.*"]
    regexes: List[str]

class BulkRelodgeModelRun(BaseModel):
    """Bulk model run relodge"""
    model_run_ids: List[str]
