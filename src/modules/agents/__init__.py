
REGISTRY = {}

from .rnn_agent import RNNAgent
from .maven_rnn_agent import MavenRNNAgent
from .rnn_v_agent import RNNVAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_v"] = RNNVAgent
REGISTRY["maven_rnn"] = MavenRNNAgent