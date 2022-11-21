from matchmaking.matchmaking import Matchmaking


class DuoMatchmaking(Matchmaking):

    def list_combat(self, agent_dict, n_matches=1):
        return [[0, 1] for _ in range(n_matches)]
