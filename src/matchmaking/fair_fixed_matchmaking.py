import random

from matchmaking.matchmaking import Matchmaking


class FairFixedMatchmaking(Matchmaking):
    def __init__(self, agent_dict, n_each_match=10):
        super(FairFixedMatchmaking, self).__init__(agent_dict)
        self.n_agents = len(self.list_id)
        self.past_match_id = 0
        self.list_combat_desired=[]
        self.n_each_match_done = 0
        for _ in range(int(n_each_match)):
            for i in range(self.n_agents):
                print(len(self.list_combat_desired))
                for j in range(i, self.n_agents):
                    if i==j:
                        continue
                    self.list_combat_desired.append([i, j])
        for _ in range(int(n_each_match)):
            for i in range(self.n_agents):
                for j in range(i, self.n_agents):
                    if i==j:
                        continue
                    self.list_combat_desired.append([j, i])
        self.done=False
        random.shuffle(self.list_combat_desired)

    def add_failed_combat(self, list_episode_matches, win_list):
        for idx, match in enumerate(list_episode_matches):
            if win_list[idx] is None or win_list[idx][0] is None or win_list[idx][1] is None:
                print("Fail", match)
                self.list_combat_desired.append(match)

    def list_combat(self, agent_dict, n_matches):
        if len(self.list_combat_desired) % 100 == 0:
            print("remaining", len(self.list_combat_desired))

        if self.done or len(self.list_combat_desired) == 0:
            return None
        to_ret=[]
        for _ in range(n_matches):
            try:
                to_ret.append(self.list_combat_desired.pop())
            except IndexError:
                to_ret.append([10, 10]) # Heuristic
                self.done=True
        return to_ret