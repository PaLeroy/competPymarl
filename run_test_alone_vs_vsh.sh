#!/usr/bin/env bash
source env/bin/activate
set -x
file1="path_to_models/population_alone_$2_$1/agent_id_0"
file2="path_to_models/population_$2_vs_heuristic_$1/agent_id_0"
name_var2="population_test_vs_heuristic_$2_vs_heuristic_$1"
name_var1="population_test_vs_heuristic_alone_$2_$1"

timeout 5m python3 src/main_popu_to_test.py \
--config=popu_$2_vs_heuristic_test --env-config=sc2_compet \
with 'env_args.map_name=3m_compet' \
'env_args.log_more_stats=True' \
'name='$name_var1 \
'agent_type_1.checkpoint_path=["'$file1'"]'

sleep 10
pkill python
pkill Main_Thread
timeout 5m python3 src/main_popu_to_test.py \
--config=popu_$2_vs_heuristic_test --env-config=sc2_compet \
with 'env_args.map_name=3m_compet' \
'env_args.log_more_stats=True' \
'name='$name_var2 \
'agent_type_1.checkpoint_path=["'$file2'"]'

sleep 10
pkill python
pkill Main_Thread
deactivate
