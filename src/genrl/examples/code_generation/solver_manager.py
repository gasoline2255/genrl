from genrl.game import BaseGameManager
from genrl.data import DataManager
from genrl.rewards import RewardManager
from genrl.trainer import TrainerModule
from genrl.communication import Communication
from genrl.roles import RoleManager
from genrl.state import GameState

class CodeGenerationGameManager(BaseGameManager):
    def __init__(
        self,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        self.data_manager.initialize(self.communication)

    def _hook_after_rewards_updated(self):
        for stage in range(self.state.stage):
            root_state = self.state.get_stage_state(stage)
            self.data_manager.send_response(self.rewards[stage], root_state)
                        