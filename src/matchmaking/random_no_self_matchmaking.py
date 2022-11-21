from matchmaking.matchmaking import Matchmaking
import numpy as np


class RandomNoSelfMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        return [np.random.choice(self.list_id, 2, replace=False).tolist() for i in range(n_matches)]