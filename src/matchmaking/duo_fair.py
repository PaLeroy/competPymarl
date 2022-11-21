from matchmaking.matchmaking import Matchmaking


class DuoFair(Matchmaking):

    def list_combat(self, agent_dict, n_matches):
        assert n_matches % 2 == 0

        return [[0, 1] for _ in range(int(n_matches/2))]\
               + [[1, 0] for _ in range(int(n_matches/2))]