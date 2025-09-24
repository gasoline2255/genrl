from dataclasses import dataclass
from genrl.examples.code_generation.proposer import Proposer
import logging
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

    def insert_proposal(self, proposer_model: str, proposal_question: str, proposal_tests: str, proposal_raw: str):
        obj = {
            "proposer_model": proposer_model,
            "proposal_question": proposal_question,
            "proposal_tests": proposal_tests,
            "proposal_raw": proposal_raw,
        }
        self.backend.put(obj, sub_key="proposer".encode())

    def request_training_data(self, train_batch_size: int):
        data = []
        while len(data) < train_batch_size:
            obj_ = self.backend.get(sub_key="solver".encode())
            obj = list(obj_.values())
            obj = [sample for sample in obj if sample['dataset'] == 'proposer']
            data.extend(obj)

        return data


def insert(proposer_client: ProposerClientDHT, proposer: Proposer, config: ProposerServiceConfig):
    try:
        model_name = proposer.model.name_or_path
    except AttributeError:
        model_name = "none"

    for _ in range(config.num_proposals):
        proposal, proposal_raw = proposer.generate_proposal()
        proposer_client.insert_proposal(model_name, proposal["question"], proposal["tests"], proposal_raw)
        logger.info(f"Proposal inserted")


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

    logger.info(f"Training with {len(rewards)} sessions and {len(proposals)} proposals")

    proposer.train(rewards, proposals)
    logger.info(f"Training completed")


def main():
    config = ProposerServiceConfig(model="Qwen/Qwen3-0.6B", num_proposals=1, batch_size=3)    

    proposer = Proposer(config.model)
    backend = HivemindBackend()
    proposer_client = ProposerClientDHT(backend)
    while True:
        insert(proposer_client, proposer, config)
        train(proposer_client, proposer, config)
   

if __name__ == "__main__":
    main()