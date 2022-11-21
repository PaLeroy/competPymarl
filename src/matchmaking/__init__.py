from matchmaking.duo_matchmaking import DuoMatchmaking
from matchmaking.duo_random_matchmaking import DuoRandomMatchmaking
from matchmaking.random_diff_matchmaking import RandomDiffMatchmaking
from matchmaking.random_matchmaking import RandomMatchmaking
from matchmaking.single_matchmaking import SingleMatchmaking
from matchmaking.random_no_self_matchmaking import RandomNoSelfMatchmaking
from matchmaking.duo_fair import DuoFair
from matchmaking.fair_fixed_matchmaking import FairFixedMatchmaking

REGISTRY = {"duo": DuoMatchmaking,
            "duo_fair": DuoFair,
            "fair_fixed_elo": FairFixedMatchmaking,
            "single": SingleMatchmaking,
            "duo_random": DuoRandomMatchmaking,
            "random": RandomMatchmaking,
            "random_test_elo": RandomNoSelfMatchmaking,
            "random_diff": RandomDiffMatchmaking}

