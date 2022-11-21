from .qv_learner import QVLearner
from .do_not_learn import DoNotLearn
from .iac_learner import IACLearner
from .maven_learner import MavenLearner
from .q_leaner_exec import QLearnerExec
from .q_learner import QLearner
from .coma_learner import COMALearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qv_learner"] = QVLearner
REGISTRY["q_learner_exec"] = QLearnerExec
REGISTRY["coma_learner"] = COMALearner
REGISTRY["iac_learner"] = IACLearner
REGISTRY["do_not_learn"] = DoNotLearn
REGISTRY["maven_learner"] = MavenLearner
