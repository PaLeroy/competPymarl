#!/usr/bin/env bash
source env/bin/activate
set -x
file1="path_to_models/population_5_$1_$2/agent_id_$3"
file2="path_to_models/population_5_$4_$5/agent_id_$6"
name_var="population_test_duo_popu_$1_$2_$3_vs_$4_$5_$6"
timeout 15m python3 src/main_popu_to_test.py \
--config=popu_duo_$3_test --env-config=sc2_compet \
with 'env_args.map_name=3m_compet' \
'env_args.log_more_stats=True' \
'name='$name_var \
'agent_type_1.checkpoint_path=["'$file1'", "'$file2'"]'
deactivate

sleep 10
pkill python
pkill Main_Thread
