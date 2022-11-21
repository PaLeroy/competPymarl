from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class DoNothingMAC:
    def __init__(self, scheme, groups, args):
        self.args = args

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        chosen_actions = []
        for avail_actions_ in avail_actions:
            chosen_actions_ = []
            for avail_act in avail_actions_:
                if avail_act[0] == 0:
                    chosen_actions_.append(1)
                else:
                    chosen_actions_.append(0)
            chosen_actions.append(chosen_actions_)
        if self.args.use_cuda:
            chosen_actions = th.cuda.LongTensor(chosen_actions)
        else:
            chosen_actions = th.LongTensor(chosen_actions)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        pass

    def init_hidden(self, batch_size):
        pass

    def parameters(self):
        pass

    def load_state(self, other_mac):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def _build_agents(self, input_shape):
        pass

    def _build_inputs(self, batch, t):
        pass

    def _get_input_shape(self, scheme):
        pass
