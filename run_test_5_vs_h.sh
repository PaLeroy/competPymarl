#!/usr/bin/env bash
source env/bin/activate
#set -x
folder_path=$1
folder_name=$(basename $folder_path)
algo="${folder_name%%__2021*}"
algo="${algo##population_5_}"
echo $algo

for (( i = 0; i < 5; i++ )); do
    file="$folder_path/agent_id_"$i
    name=$folder_name"_test_vs_h_agent_id_"$i
    echo $file
    echo $name
    timeout 15m python3 src/main_popu_to_test.py \
    --config=popu_"$algo"_vs_heuristic_test --env-config=sc2_compet \
    with 'env_args.map_name=3m_compet' \
    'env_args.log_more_stats=True' \
    'name='$name \
    'agent_type_1.checkpoint_path=["'$file'"]'
    sleep 10
    pkill python
    pkill Main_Thread
done

deactivate