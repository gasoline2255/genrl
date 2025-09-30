from dataclasses import dataclass
from genrl.examples.code_generation.proposer import Proposer
import logging
import random
from genrl.communication.hivemind.hivemind_backend import HivemindBackend, HivemindRendezvouz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProposerServiceConfig:
    model: str
    num_proposals: int
    batch_size: int
    initial_peers: list[str]=None


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


def insert(proposer_client: ProposerClientDHT, proposer: Proposer, config: ProposerServiceConfig):
    try:
        model_name = proposer.model.name_or_path
    except AttributeError:
        model_name = "none"
    proposals = []
    for _ in range(config.num_proposals):
        proposal = proposer.generate_proposal()
        proposals.append(proposal)
    proposer_client.insert_proposal(model_name, proposals)
    logger.info(f"{len(proposals)} proposals inserted")


def train(proposer_client: ProposerClientDHT, proposer: Proposer, config: ProposerServiceConfig):

    training_data = proposer_client.request_training_data(config.batch_size)
    if len(training_data) == 0:
        logger.info("No training data found")
        return
    elif len(training_data) > config.batch_size:
        logger.info("Training data is larger than batch size")
        training_data = training_data[:config.batch_size]
        
    rewards = []
    proposals = []
    for sample in training_data:
        rewards.append(sample["reward"])
        proposals.append(sample["proposal_raw"])


    if len(rewards) == 0:
        logger.info("No training data found")
        return

    logger.info(f"Training with {len(proposals)} proposals")

    proposer.train(rewards, proposals)
    logger.info(f"Training completed")


def main():
    config = ProposerServiceConfig(model="Qwen/Qwen3-4B-Instruct-2507", num_proposals=3, batch_size=3)    

    proposer = Proposer(config.model)
    backend = HivemindBackend()
    proposer_client = ProposerClientDHT(backend)
    while True:
        insert(proposer_client, proposer, config)
        train(proposer_client, proposer, config)
   

if __name__ == "__main__":
    HivemindRendezvouz().init(is_master=True)
    main()