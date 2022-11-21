from matchmaking.matchmaking import Matchmaking
import numpy as np


class RandomDiffMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        return np.random.choice(self.list_id, (n_matches, 2),
                                replace=False).tolist()
