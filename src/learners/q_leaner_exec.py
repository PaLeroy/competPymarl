import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearnerExec:
    def __init__(self, mac, scheme, logger, args, id_agent=""):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.id_agent = id_agent if id_agent == "" else "agent_id_" + id_agent + "_"
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        pass

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        if self.id_agent == "":
            self.logger.console_logger.info("Updated target network")
        else:
            self.logger.console_logger.info(
                "Updated target network of agent_id=" + str(
                    self.id_agent[:-1]))

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
