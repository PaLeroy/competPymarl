import numpy as np
from copy import deepcopy

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, RunObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
from utils.logging import get_logger
import yaml

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from matchmaking import REGISTRY as m_REGISTRY

from components.episode_buffer import ReplayBuffer, ReplayBufferPopulation
from components.transforms import OneHot

from main import _get_config, recursive_dict_update
from run import args_sanity_check, evaluate_sequential

SETTINGS[
    'CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log, env_args):
    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]

    # run the framework
    run_test(_run, _config, _log)


def run_test(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = args.unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))),
                                     "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_population_test(args=args, logger=logger)
    time.sleep(300)  # To let sacred fileobserver write everything

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=60)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_population_test(args, logger):
    # Creation of the agents dictionary
    agent_dict = {}
    agent_id = 0
    for i in range(args.n_agent_type):
        args_this_agent = args.__dict__["agent_type_" + str(i + 1)]
        n_agent_this_type = args_this_agent['number']

        assert n_agent_this_type == len(args_this_agent["save_model"])
        save_model = args_this_agent["save_model"]

        assert n_agent_this_type == len(args_this_agent["save_model_interval"])
        save_model_interval = args_this_agent["save_model_interval"]
        assert n_agent_this_type == len(args_this_agent["checkpoint_path"])
        checkpoint_path = args_this_agent["checkpoint_path"]

        assert n_agent_this_type == len(args_this_agent["load_step"])
        load_step = args_this_agent["load_step"]

        for j in range(n_agent_this_type):
            args_this_agent_modified = args_this_agent.copy()
            args_this_agent_modified["save_model"] = save_model[j]
            args_this_agent_modified["save_model_interval"] = \
                save_model_interval[j]
            args_this_agent_modified["checkpoint_path"] = checkpoint_path[j]
            args_this_agent_modified["load_step"] = load_step[j]
            args_this_agent_modified["batch_size_run"] = args.batch_size_run
            new_agent = {
                'id': agent_id,
                'args_sn': SN(**args_this_agent_modified)
            }
            agent_dict[agent_id] = new_agent
            agent_id += 1

    runner = r_REGISTRY[args.runner](args=args, logger=logger,
                                     agent_dict=agent_dict)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    for k, v in agent_dict.items():
        # noinspection DuplicatedCode
        scheme = {"state": {"vshape": env_info["state_shape"]},
                  "obs": {"vshape": env_info["obs_shape"][0],
                          "group": "agents"},
                  "actions": {"vshape": (1,), "group": "agents",
                              "dtype": th.long},
                  "avail_actions": {"vshape": (env_info["n_actions"],),
                                    "group": "agents", "dtype": th.int},
                  "reward": {"vshape": (1,)},
                  "terminated": {"vshape": (1,), "dtype": th.uint8},
                  }
        if agent_dict[k]['args_sn'].mac == "maven_mac":
            scheme["noise"] = {"vshape": (agent_dict[k]['args_sn'].noise_dim,)}

        agent_dict[k]['scheme_buffer'] = scheme

    groups = {
        "agents": args.n_agents,
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBufferPopulation(groups,
                                    args.buffer_size,
                                    env_info["episode_limit"] + 1,
                                    agent_dict,
                                    preprocess=preprocess,
                                    device="cpu"
                                    if args.buffer_cpu_only else args.device)

    for k, v in agent_dict.items():
        agent_dict[k]['args_sn'].n_agents = env_info["n_agents"]
        agent_dict[k]['args_sn'].n_actions = env_info["n_actions"]
        agent_dict[k]['args_sn'].state_shape = env_info["state_shape"]
        agent_dict[k]['args_sn'].use_cuda = args.use_cuda
        agent_dict[k]['mac'] = mac_REGISTRY[agent_dict[k]['args_sn'].mac](
            buffer.scheme,
            groups,
            agent_dict[k]['args_sn'])
        agent_dict[k]['learner'] \
            = le_REGISTRY[agent_dict[k]['args_sn'].learner](
            agent_dict[k]['mac'],
            buffer.scheme,
            logger, agent_dict[k]['args_sn'], id_agent=str(k))
        agent_dict[k]['t_total'] = 0
        agent_dict[k]['episode'] = 0
        agent_dict[k]['model_save_time'] = 0

        if args.use_cuda:
            agent_dict[k]['learner'].cuda()

        checkpoint_path_ = agent_dict[k]['args_sn'].checkpoint_path
        agent_dict[k]["load_timesteps"] = []
        if checkpoint_path_ != "":
            if not os.path.isdir(checkpoint_path_):
                logger.console_logger.info(
                    "Checkpoint directory {} doesn't exist".format(
                        checkpoint_path_))
                return
            # Go through all files in args.checkpoint_path
            for name in os.listdir(checkpoint_path_):
                full_name = os.path.join(checkpoint_path_, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    agent_dict[k]["load_timesteps"].append(int(name))

        elif agent_dict[k]['args_sn'].mac == "do_not_mac":
            pass
        else:
            logger.console_logger.info("Checkpoint directory doesn't exist")
            exit()

    match_maker = m_REGISTRY[args.matchmaking](agent_dict)

    if args.matchmaking == "duo_fair":
        # Only 2 agents
        agent_dict[0]["load_timesteps"] = sorted(
            agent_dict[0]["load_timesteps"])
        agent_dict[1]["load_timesteps"] = sorted(
            agent_dict[1]["load_timesteps"])
        print("load_timesteps_team_1=", agent_dict[0]["load_timesteps"])
        print("load_timesteps_team_2=", agent_dict[1]["load_timesteps"])
        if len(agent_dict[1]["load_timesteps"]) > 0:
            load_factor_team1 = 0
            load_factor_team2 = 0
            idx_team1 = 0
            idx_team2 = 0
            time_steps_team1 = agent_dict[0]["load_timesteps"]
            time_steps_team2 = agent_dict[1]["load_timesteps"]
            while idx_team1 < len(time_steps_team1) and idx_team2 < len(time_steps_team2):
                timestep_to_load1 = time_steps_team1[idx_team1]
                while idx_team1 < len(
                        time_steps_team1) and timestep_to_load1 < load_factor_team1 * args.load_timesteps_spacing:
                    timestep_to_load1 = time_steps_team1[idx_team1]
                    idx_team1 += 1
                else:
                    load_factor_team1 += 1

                timestep_to_load2 = time_steps_team2[idx_team2]
                while idx_team2 < len(
                        time_steps_team2) and timestep_to_load2 < load_factor_team2 * args.load_timesteps_spacing:
                    timestep_to_load2 = time_steps_team2[idx_team2]
                    idx_team2 += 1
                else:
                    load_factor_team2 += 1
                runner.setup(agent_dict=agent_dict, groups=groups,
                             preprocess=preprocess)
                print("timestep_to_load", timestep_to_load1, timestep_to_load2)
                model_path1 = os.path.join(
                    agent_dict[0]['args_sn'].checkpoint_path,
                    str(timestep_to_load1))
                logger.console_logger.info(
                    "Loading model from {}".format(model_path1))
                agent_dict[0]['learner'].load_models(model_path1)
                agent_dict[0]['t_total'] = timestep_to_load1
                runner.load_models(model_path1, 0, agent_dict)

                model_path2 = os.path.join(
                    agent_dict[1]['args_sn'].checkpoint_path,
                    str(timestep_to_load2))
                logger.console_logger.info(
                    "Loading model from {}".format(model_path2))
                agent_dict[1]['learner'].load_models(model_path2)
                agent_dict[1]['t_total'] = timestep_to_load2
                runner.load_models(model_path2, 1, agent_dict)

                cur_tested = 0
                while cur_tested < args.test_nepisode:
                    # Run for a whole episode at a time
                    list_episode_matches = match_maker.list_combat(agent_dict,
                                                                   n_matches=args.batch_size_run)
                    runner.setup_agents(list_episode_matches, agent_dict)
                    episode_batches, total_times, win_list = runner.run(
                        test_mode=True)
                    cur_tested += len(win_list)
        else:
            load_factor_team1 = 0
            for idx_, timestep_to_load1 in enumerate(
                    agent_dict[0]["load_timesteps"]):
                if timestep_to_load1 < load_factor_team1 * args.load_timesteps_spacing:
                    continue
                else:
                    load_factor_team1 += 1
                runner.setup(agent_dict=agent_dict, groups=groups,
                             preprocess=preprocess)

                print("timestep_to_load", timestep_to_load1, timestep_to_load1)
                model_path1 = os.path.join(
                    agent_dict[0]['args_sn'].checkpoint_path,
                    str(timestep_to_load1))
                logger.console_logger.info(
                    "Loading model from {}".format(model_path1))
                agent_dict[0]['learner'].load_models(model_path1)
                agent_dict[0]['t_total'] = timestep_to_load1
                runner.load_models(model_path1, 0, agent_dict)

                model_path2 = os.path.join(
                    agent_dict[1]['args_sn'].checkpoint_path,
                    str(timestep_to_load1))
                logger.console_logger.info(
                    "Loading model from {}".format(model_path2))
                agent_dict[1]['learner'].load_models(model_path2)
                agent_dict[1]['t_total'] = timestep_to_load1

                cur_tested = 0
                while cur_tested < args.test_nepisode:
                    # Run for a whole episode at a time
                    list_episode_matches = match_maker.list_combat(agent_dict,
                                                                   n_matches=args.batch_size_run)
                    runner.setup_agents(list_episode_matches, agent_dict)
                    episode_batches, total_times, win_list = runner.run(
                        test_mode=True)
                    cur_tested += len(win_list)
    elif args.matchmaking == "random_test_elo" or "fair_fixed_elo":
        # unknown number of agents.
        # load args.loadstep timing.
        # first find the timesteps closest to the load step asked.
        runner.setup(agent_dict=agent_dict, groups=groups,
                     preprocess=preprocess)
        for agent_id, agent_info in agent_dict.items():
            load_step = agent_info["args_sn"].load_step
            agent_dict[agent_id]['t_total_test'] = 0
            if agent_info["args_sn"].learner != "do_not_learn":
                if load_step <= 0:
                    logger.console_logger.info("Unspecified load_step")
                    exit()
                else:
                    timestep_to_load = min(v for v in agent_info["load_timesteps"] if v >= load_step)
                    model_path = os.path.join(agent_dict[agent_id]['args_sn'].checkpoint_path, str(timestep_to_load))
                    agent_dict[agent_id]["model_name"] \
                        = str(model_path.split("/")[-3]) + "_" + str(model_path.split("/")[-2])

                    logger.console_logger.info(
                        "Loading model from {}".format(model_path))
                    agent_dict[agent_id]['learner'].load_models(model_path)

                    runner.load_models(model_path, agent_id, agent_dict)
            else:
                agent_dict[agent_id]["model_name"] = "heuristic" + str(agent_id)
        cur_tested = 0
        while True:
            # Run for a whole episode at a time
            list_episode_matches = match_maker.list_combat(agent_dict,
                                                           n_matches=args.batch_size_run)
            if list_episode_matches == None:
                break

            runner.setup_agents(list_episode_matches, agent_dict)
            episode_batches, total_times, win_list = runner.run(
                test_mode=True)
            agent_that_played = []
            for idx, match in enumerate(list_episode_matches):
                for agent_id in match:
                    agent_that_played.append(agent_id)
                    if win_list[idx] is not None:
                        agent_dict[agent_id]['t_total_test'] += 1
            match_maker.add_failed_combat(list_episode_matches, win_list)
            match_maker.update_elo(agent_dict, list_episode_matches, win_list)
            match_maker.update_ranking(agent_dict)
            cur_tested += len(win_list)
            if cur_tested % 100 == 0:
                logger.console_logger.info(
                    "Already {} played".format(cur_tested))
            for k, v in agent_dict.items():
                if k not in agent_that_played:
                    continue
                logger.log_stat("elo_" + str(agent_dict[k]["model_name"]),
                                agent_dict[k]["elo"], agent_dict[k]['t_total_test'])
                logger.log_stat("ranking_" + str(agent_dict[k]["model_name"]),
                                agent_dict[k]["ranking"], agent_dict[k]['t_total_test'])

        print("------------ fin ------------")
        for k, v in agent_dict.items():
            print("agent:", str(k).ljust(2),
                  " elo:", str(agent_dict[k]["elo"]).ljust(20),
                  " position:", str(agent_dict[k]["ranking"]).ljust(3),
                  " type:", agent_dict[k]["args_sn"].learner.ljust(15))
            # logger.log_stat("elo_" + str(agent_dict[k]["model_name"]),
            #                 agent_dict[k]["elo"], agent_dict[k]['t_total_test'])
            # logger.log_stat("ranking_" + str(agent_dict[k]["model_name"]),
            #                 agent_dict[k]["ranking"], agent_dict[k]['t_total_test'])
    # if args.matchmaking == "single":
    #     agent_dict[0]["load_timesteps"] = sorted(
    #         agent_dict[0]["load_timesteps"])
    #     cptaze = args.n_skip
    #     for idx_, timestep_to_load in enumerate(
    #             agent_dict[0]["load_timesteps"]):
    #         # if timestep_to_load < 7500000:
    #         #     continue
    #         if cptaze != args.n_skip:
    #             cptaze += 1
    #             continue
    #         else:
    #             cptaze = 0
    #         print("timestep_to_load", timestep_to_load)
    #         model_path = os.path.join(agent_dict[0]['args_sn'].checkpoint_path,
    #                                   str(timestep_to_load))
    #
    #         logger.console_logger.info(
    #             "Loading model from {}".format(model_path))
    #         agent_dict[0]['learner'].load_models(model_path)
    #         agent_dict[0]['t_total'] = timestep_to_load
    #
    #         runner.setup(scheme=scheme_buffer, groups=groups,
    #                      preprocess=preprocess)
    #
    #         for _ in range(args.n_epsiode_per_test):
    #             # Run for a whole episode at a time
    #             list_episode_matches = match_maker.list_combat(agent_dict,
    #                                                            n_matches=args.batch_size_run)
    #             runner.setup_agents(list_episode_matches, agent_dict)
    #             # episode_batches, total_times, win_list = runner.run(
    #             #     test_mode=True)
    #             # print(win_list)
    #
    # if args.matchmaking == "duo":
    #     agent_dict[0]["load_timesteps"] = sorted(
    #         agent_dict[0]["load_timesteps"])
    #     for idx_, timestep_to_load in enumerate(
    #             agent_dict[0]["load_timesteps"]):
    #         # if timestep_to_load < 4000000:
    #         #     continue
    #         print("timestep_to_load", timestep_to_load)
    #         model_path1 = os.path.join(
    #             agent_dict[0]['args_sn'].checkpoint_path,
    #             str(timestep_to_load))
    #         logger.console_logger.info(
    #             "Loading model from {}".format(model_path1))
    #         agent_dict[0]['learner'].load_models(model_path1)
    #         agent_dict[0]['t_total'] = timestep_to_load
    #
    #         model_path2 = os.path.join(
    #             agent_dict[1]['args_sn'].checkpoint_path,
    #             str(timestep_to_load))
    #         logger.console_logger.info(
    #             "Loading model from {}".format(model_path2))
    #         agent_dict[1]['learner'].load_models(model_path2)
    #         agent_dict[1]['t_total'] = timestep_to_load
    #
    #
    #         runner.setup(scheme=scheme_buffer, groups=groups,
    #                      preprocess=preprocess)
    #
    #         for _ in range(args.n_epsiode_per_test):
    #             # Run for a whole episode at a time
    #             list_episode_matches = match_maker.list_combat(agent_dict,
    #                                                            n_matches=args.batch_size_run)
    #             runner.setup_agents(list_episode_matches, agent_dict)
    #             episode_batches, total_times, win_list = runner.run(
    #                 test_mode=True)
    #             print(win_list)
    else:
        logger.console_logger.info("Unknown matchmaking")
        exit()

    runner.close_env()
    logger.console_logger.info("Finished Testing")


def check_for_name(params):
    for param in params:
        if param.startswith("name="):
            return param[5:]
    return None


class SetID(RunObserver):
    priority = 50  # very high priority

    def __init__(self, custom_id):
        self.custom_id = custom_id

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        return self.custom_id  # started_event should returns the _id


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
            os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
            "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    name = check_for_name(params)
    if name is not None:
        config_dict['name'] = name
    config_dict["unique_token"] = "{}__{}".format(config_dict['name'], datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(SetID(config_dict["unique_token"]))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
