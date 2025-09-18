import os
import tempfile
import time

import torch.multiprocessing as mp
import pytest 
# CI can't do this, Skip this entire test file
pytest.skip("Skipping all tests in test_hivemind_backend.py", allow_module_level=True)

from genrl.communication.hivemind.hivemind_backend import (
    HivemindBackend,
    HivemindRendezvouz,
)


def _test_hivemind_backend(rank):
    HivemindRendezvouz.init(is_master=rank == 0)
    backend = HivemindBackend(timeout=5)

    if rank == 0:
        obj = "this is text"
        backend.put(obj, sub_key="proposer".encode())
        obj_ = backend.get(sub_key="solver".encode())
        obj = list(obj_.values())[0]
        assert obj == "this is text from solver"
    else:
        obj_ = backend.get(sub_key="proposer".encode())
        obj = list(obj_.values())[0] + " from solver"
        assert obj == "this is text from solver"
        backend.put(obj, sub_key="solver".encode())
    time.sleep(5)


def test_hivemind_backend():
    os.environ["HIVEMIND_WORLD_SIZE"] = "2"
    os.environ["LAMBDA"] = "1"
    with tempfile.NamedTemporaryFile() as f:
        os.environ["MASTER_FILE"] = f.name
        mp.spawn(
            _test_hivemind_backend,
            args=(),
            nprocs=2,
            join=True,
            daemon=False,
        )
