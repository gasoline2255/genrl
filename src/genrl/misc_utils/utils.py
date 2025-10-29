import hashlib
from typing import Any


def generate_md5_hash_id(hashable_obj: Any) -> str:
    hash_fxn = hashlib.md5()
    hash_fxn.update(str.encode(hashable_obj))
    return hash_fxn.hexdigest()
    