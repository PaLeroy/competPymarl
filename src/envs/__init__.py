from functools import partial

import sys
import os
from smac.env.multiagentenv import MultiAgentEnv
from smac.env.starcraft2.starcraft2 import StarCraft2Env
from smac.env.starcraft2.compet_starcraft2 import CompetStarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2compet"] = partial(env_fn, env=CompetStarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
