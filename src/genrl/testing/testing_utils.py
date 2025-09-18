from genrl.communication.communication import Communication
from genrl.logging_utils.global_defs import get_logger


class TestGameManager:
    def __init__(self, msg: str, comm: Communication | None = None):
        self.msg = msg
        self.comm = comm

    def run_game(self):
        if self.comm is not None:
            gathered_obj = self.comm.all_gather_object([1])
            get_logger().info(f"Run backend gather with output: {gathered_obj}")
        get_logger().info(f"Run game with message: {self.msg}")


class TestTwoRoleGameManager:
    def __init__(self, msg: str, comm: Communication | None = None):
        self.msg = msg
        self.comm = comm

    def run_game(self):
        if self.comm is not None:
            peer_id = self.comm.get_id()
            if isinstance(peer_id, str):
                role = int.from_bytes(peer_id.encode(), byteorder='big') % 2
            elif isinstance(peer_id, bytes):
                role = int.from_bytes(peer_id, byteorder='big') % 2
            else:
                role = peer_id % 2

            if role == 0:
                data = [self.msg]
                self.comm.put(data, sub_key="proposer".encode())
                solutions = self.comm.get(sub_key="solver".encode())
                print(solutions)
            else:
                problems = self.comm.get(data, sub_key="proposer".encode())
                solutions = [
                    x + ", this is solution" for x in problems
                ]
                self.comm.put(solutions, sub_key="solver".encode())

        get_logger().info(f"Finished game.")
