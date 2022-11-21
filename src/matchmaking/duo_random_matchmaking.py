from matchmaking.matchmaking import Matchmaking
import numpy as np


class DuoRandomMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        rand_ = np.random.random_sample()
        if rand_ < 0.5:
            return [(0, 1), ]
        else:
            return [(1, 0), ]
