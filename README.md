# Competitive pymarl

This is the code of our paper entilted [Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition](https://arxiv.org/abs/2211.11886).



This code is a modification of the [former pymarl](https://github.com/oxwhirl/pymarl) to allow to train several teams at the same time in a competitive mode in the [competitive SMAC](https://github.com/paleroy/competSmac) environment.


## Installation instructions

We hereafter consider an installation with python virtual environments.
If you consider using an other environment manager such as conda, consider to modify the following scripts.

Install the python virtual environment (**python 3.6** required):

```shell script
./install_venv.sh
```


Set up StarCraft II(on Linux). Check [competitive SMAC](https://github.com/paleroy/competSmac) for more details:

```shell
bash install_sc2.sh
```

Set up SMAC Maps:

```shell
cp -r src/envs/starcraft2/maps/SMAC_Maps/ 3rdparty/StarCraftII/Maps/
```


## Testing sc2 installation
Execute the following command that play randomly at competitive SMAC to test your sc2 installation.

```shell script
source env/bin/activate
python3.6 -m smac.examples.random_agents_compet.py
deactivate
```

# Training instructions
You can train nine types of team with the following command:
You can change the map by modifying the map_name from '3m_compet' to '3s5z_compet'.

```shell script
source env/bin/activate
python3.6 src/main.py --config="config_name" --env-config=sc2_compet with env_args.map_name=3m_compet
deactivate
```

The parameter "config_name" defines which type of team you will train.
Config files are in the folder "/src/config/algs".
Here is the list of train config that are self-explanatory:

- `popu_qmix_vs_heuristic`
- `popu_qmix_self`
- `popu_qmix_5`
- `popu_qvmix_vs_heuristic`
- `popu_qvmix_self`
- `popu_qvmix_5`
- `popu_maven_vs_heuristic`
- `popu_maven_self`
- `popu_maven_5`

# Testing instructions
Once trained, it is possible to test your teams in different configurations.

We provide all the test scripts executed to compute Elo scores after training or win-rates along training.

For the Elo, scripts have a name prefixed "elo_".
They will automatically find the 10000000th saved networks and execute the experiment as described in the paper.

For the win-rates, scripts have a name prefixed "run_test".

All those scripts sometimes require arguments that are self-explanatory.

# Heuristic
The heuristic is implemented in the [competitive SMAC](https://github.com/paleroy/competSmac) and can be modified.


# Citing competitive PYMARL 

If you use the competitive PYMARL implementation in your own work, please cite our paper: [Value-based CTDE Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition](https://arxiv.org/abs/2211.11886).

```tex
@inproceedings{leroy2022twoteam,
title={Value-based {CTDE} Methods in Symmetric Two-team Markov Game: from Cooperation to Team Competition},
author={Leroy, Pascal and Pisane, Jonathan and Ernst, Damien},
booktitle={Deep Reinforcement Learning Workshop NeurIPS 2022},
year={2022}
}
```
