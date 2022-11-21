import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class DoNotLearn:
    def __init__(self, mac, scheme, logger, args, id_agent=""):
        self.args = args
        self.logger = logger

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        pass

    def _update_targets(self):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
