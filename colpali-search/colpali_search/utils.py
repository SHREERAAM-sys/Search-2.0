from datetime import datetime, timezone
from typing import List


def get_now():
    return datetime.now(timezone.utc).isoformat()


def convert_tensors_to_list_of_lists(tensors: List) -> List[List[float]]:
    return [tensor.view(-1).tolist() for tensor in tensors]
