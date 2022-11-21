import datetime
import os
import pprint
import time
import threading
import torch as th
import multiprocessing

from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from matchmaking import REGISTRY as m_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.episode_buffer import ReplayBufferPopulation

num_cores = multiprocessing.cpu_count()

def run(_run, _config, _log):
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
    if hasattr(args, 'runner_function'):
        if args.runner_function == 'population':
            run_population(args=args, logger=logger)
    else:
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)




def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_population(args, logger):
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
    # Take care that env info is made for 2 teams (-> obs is a tuple)

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
        if checkpoint_path_ != "":
            timesteps = []

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
                    timesteps.append(int(name))

            if agent_dict[k]['args_sn'].load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps,
                                       key=lambda x: abs(x - agent_dict[k][
                                           'args_sn'].load_step))

            model_path = os.path.join(checkpoint_path_,
                                      str(timestep_to_load))

            logger.console_logger.info(
                "Loading model from {}".format(model_path))
            agent_dict[k]['learner'].load_models(model_path)
            agent_dict[k]['t_total'] = timestep_to_load

            if args.evaluate or args.save_replay:
                evaluate_sequential(args, runner)
                return
    match_maker = m_REGISTRY[args.matchmaking](agent_dict)

    runner.setup(agent_dict=agent_dict, groups=groups, preprocess=preprocess)

    if args.use_cuda:
        runner.cuda()

    # start training
    last_test_T = -args.test_interval - 1
    last_log_T = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))

    min_played_times = 0
    while min_played_times <= args.t_max:
        # Run for a whole episode at a time
        list_episode_matches = match_maker.list_combat(agent_dict,
                                                       n_matches=args.batch_size_run)
        runner.setup_agents(list_episode_matches, agent_dict)
        episode_batches, total_times, win_list = runner.run(test_mode=False)
        match_maker.update_elo(agent_dict, list_episode_matches, win_list)
        for idx_, match in enumerate(list_episode_matches):
            if total_times[idx_] is None:
                continue
            for idx2_, agent_id in enumerate(match):
                agent_dict[agent_id]['t_total'] += total_times[idx_][idx2_]
                agent_dict[agent_id]['episode'] += 1
        buffer.insert_episode_batch(episode_batches, agent_dict,
                                    list_episode_matches)

        played_times = [v['t_total'] for _, v in agent_dict.items()]
        min_played_times = min(played_times)
        list_agent_can_sample = buffer.can_sample(agent_dict)
        # update only the ones that just played
        list_agent_can_sample = list(set([j for i in list_episode_matches for j in i if j in list_agent_can_sample]))

        if list_agent_can_sample:
            for agent_id in list_agent_can_sample:
                # Train agents that can be trained
                episode_sample = buffer.sample(agent_id, agent_dict)
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                agent_dict[agent_id]['learner'].train(episode_sample,
                                                      agent_dict[agent_id][
                                                          't_total'],
                                                      agent_dict[agent_id][
                                                          'episode'])
        for agent_id, dict___ in agent_dict.items():
            if dict___['args_sn'].save_model \
                    and (dict___['t_total'] - dict___['model_save_time']
                         >= dict___['args_sn'].save_model_interval
                         >= dict___['args_sn'].save_model_interval
                         or dict___['model_save_time'] == 0):
                agent_dict[agent_id]['model_save_time'] \
                    = agent_dict[agent_id]['t_total']

                save_path = os.path.join(args.local_results_path, "models",
                                         args.unique_token,
                                         "agent_id_" + str(agent_id),
                                         str(agent_dict[agent_id]['t_total']))

                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info(
                    "Saving models to {}".format(save_path))

                agent_dict[agent_id]['learner'].save_models(save_path)
                runner.save_models(save_path, agent_id, agent_dict)

        if (min_played_times - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(min_played_times, args.t_max))
            print("test")
            last_test_T = min_played_times
            cpt = 0
            while cpt < args.test_nepisode:
                list_episode_matches = match_maker.list_combat(agent_dict,
                                                               n_matches=args.batch_size_run)
                runner.setup_agents(list_episode_matches, agent_dict)
                episode_batches, total_times, win_list = runner.run(
                    test_mode=True)
                cpt += sum([tmp is not None for tmp in total_times])

        if (min_played_times - last_log_T) >= args.log_interval:
            # logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("time_elapsed", time.time() - start_time,
                            min_played_times)
            for k, v in agent_dict.items():
                logger.log_stat("agent_id_" + str(k) + "_elo",
                                agent_dict[k]["elo"], agent_dict[k]["t_total"])
            # logger.print_recent_stats()
            last_log_T = min_played_times


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()

    if args.multi:
        args.n_agents_team1 = env_info["n_agents"]
        args.n_agents_team2 = env_info["n_enemies"]
        args.n_agents = env_info["n_agents"] + env_info["n_enemies"]
    else:
        args.n_agents = env_info["n_agents"]

    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    if args.multi:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs_team_1": {"vshape": env_info["obs_shape"][0],
                           "group": "agents_team_1"},
            "obs_team_2": {"vshape": env_info["obs_shape"][1],
                           "group": "agents_team_2"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],),
                              "group": "agents", "dtype": th.int},
            "reward": {"vshape": (args.n_agents,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        groups = {
            "agents": args.n_agents,
            "agents_team_1": args.n_agents_team1,
            "agents_team_2": args.n_agents_team2
        }
    else:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],),
                              "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,) if not args.multi else (args.n_agents,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        groups = {
            "agents": args.n_agents
        }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    if args.mac == "maven_mac":
        scheme["noise"] = {"vshape": (args.noise_dim,)}

    buffer = ReplayBuffer(scheme, groups, args.buffer_size,
                          env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(
                    args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps,
                                   key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env,
                              args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models",
                                     args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config[
            "batch_size_run"]) * config["batch_size_run"]

    return config
