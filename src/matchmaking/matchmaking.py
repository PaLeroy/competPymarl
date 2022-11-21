import numpy as np

class Matchmaking:
    def __init__(self, agent_dict):
        self.list_id = []
        for k, v in agent_dict.items():
            self.list_id.append(k)
            agent_dict[k]["elo"] = 1000
            agent_dict[k]["ranking"] = 0

    def list_combat(self, agent_dict, n_matches=1):
        return NotImplementedError

    def update_elo(self, agent_dict, list_episode_matches, win_list):
        for idx, match in enumerate(list_episode_matches):
            if win_list[idx] is None:
                continue
            win_1 = win_list[idx][0]
            win_2 = win_list[idx][1]
            if win_1 is None or win_2 is None:
                # The match did not end
                continue
            id_team_1 = match[0]
            id_team_2 = match[1]
            if id_team_1 == id_team_2:
                continue

            elo_team_1 = agent_dict[id_team_1]["elo"]
            elo_team_2 = agent_dict[id_team_2]["elo"]
            q_1 = np.power(10, elo_team_1/400)
            q_2 = np.power(10, elo_team_2/400)
            q_t = q_1 + q_2
            e_1 = q_1 / q_t
            e_2 = q_2 / q_t
            if win_1:
                s_1 = 1
                s_2 = 0
            elif win_2:
                s_1 = 0
                s_2 = 1
            else:
                s_1 = 0.5
                s_2 = 0.5
            agent_dict[id_team_1]["elo"] = max(0, elo_team_1 + 10 * (s_1 - e_1))
            agent_dict[id_team_2]["elo"] = max(0, elo_team_2 + 10 * (s_2 - e_2))

    def add_failed_combat(self, list_episode_matches, win_list):
        return NotImplementedError

    def update_ranking(self, agent_dict):
        elo_dict = {}
        for agent_id, agent_info in agent_dict.items():
            elo_dict[agent_id]=agent_dict[agent_id]["elo"]
        elo_dict= {k: v for k, v in sorted(elo_dict.items(), key=lambda item: item[1])}
        position = len(agent_dict)
        for k,v in elo_dict.items():
            agent_dict[k]["ranking"] = position
            position-=1