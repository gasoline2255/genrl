import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from genrl.communication.distributed.torch_comm import TorchBackend


def _test_communication(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method,
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    backend = TorchBackend()
    obj = [dist.get_rank()]
    gathered_obj = backend.all_gather_object(obj)
    assert len(gathered_obj) == dist.get_world_size()
    assert gathered_obj == {i: [i] for i in range(dist.get_world_size())}
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_communication(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_communication,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


def _test_put_get(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method,
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    backend = TorchBackend()

    if rank % 2 == 0:
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

    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_put_get(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_put_get,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
