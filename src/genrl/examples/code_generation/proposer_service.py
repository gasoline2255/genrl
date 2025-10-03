from dataclasses import dataclass
from genrl.examples.code_generation.proposer import Proposer, PPOConfig, VllmConfig
import logging
import random
from genrl.communication.hivemind.hivemind_backend import HivemindBackend, HivemindRendezvouz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProposerServiceConfig:
    model: str
    num_proposals: int
    train_batch_size: int
    identity_path: str
    startup_timeout: int
    beam_size: int
    get_retries: int
    do_training: bool = False


class ProposerClientDHT:
    def __init__(self, backend: HivemindBackend):
        self.backend = backend

    def insert_proposal(self, proposer_model: str, proposals: list[dict]):
        objs = [{
            "proposer_model": proposer_model,
            "proposal_question": proposal_dict["question"],
            "proposal_tests": proposal_dict["tests"],
            "proposal_raw": proposal_dict["proposal_raw"],
        } for proposal_dict in proposals]
        self.backend.put(objs, sub_key="proposer".encode())

    def request_training_data(self, train_batch_size: int) -> list[dict]:
        data = []
        obj_ = self.backend.get(sub_key="solver".encode())

        if obj_ is None or len(obj_) == 0:
            return data

        objs = list(obj_.values())

        # Batching data so this is a nested list
        for list_of_samples in objs:
            for sample in list_of_samples:
                if sample['dataset'] == 'proposer':
                    data.append(sample)
                    
        if len(data) > train_batch_size:
            data = random.sample(data, train_batch_size)
        return data


class ProposerService:
    def __init__(self,
                 service_config: ProposerServiceConfig,
                 ppo_config: PPOConfig,
                 vllm_config: VllmConfig,
                 ):
        
        backend = HivemindBackend(
            identity_path=service_config.identity_path,
            startup_timeout=service_config.startup_timeout,
            beam_size=service_config.beam_size,
            get_retries=service_config.get_retries,
        )
        proposer_client = ProposerClientDHT(backend)
        self.proposer_client = proposer_client
        self.proposer = Proposer(service_config.model, ppo_config, vllm_config)
        logger.info(f'Propser initialized with model {service_config.model}')
        self.config = service_config
    
    def insert(self):
        try:
            model_name = self.proposer.model.name_or_path
        except AttributeError:
            model_name = "none"
        proposals = []
        for _ in range(self.config.num_proposals):
            proposal = self.proposer.generate_proposal()
            proposals.append(proposal)
        self.proposer_client.insert_proposal(model_name, proposals)
        logger.info(f"{len(proposals)} proposals inserted")


    def train(self):

        training_data = self.proposer_client.request_training_data(self.config.train_batch_size)
        if len(training_data) == 0:
            logger.info("No training data found")
            return
        elif len(training_data) > self.config.train_batch_size:
            logger.info("Training data is larger than batch size")
            training_data = training_data[:self.config.train_batch_size]
            
        rewards = []
        proposals = []
        for sample in training_data:
            rewards.append(sample["reward"])
            proposals.append(sample["proposal_raw"])


        if len(rewards) == 0:
            logger.info("No training data found")
            return

        logger.info(f"Training with {len(proposals)} proposals")

        self.proposer.train(rewards, proposals)
        logger.info(f"Training completed")


    def run(self):
        logger.info("Starting proposer service")

        while True:
            self.insert()
            if self.config.do_training:
                self.train()
   

if __name__ == "__main__":
    HivemindRendezvouz().init(is_master=True)
    main()