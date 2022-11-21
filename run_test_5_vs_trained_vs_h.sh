#!/usr/bin/env bash
source env/bin/activate
#set -x
folder_path=$1
folder_path2=path_to_models
folder_name=$(basename $folder_path)
algo="${folder_name}"
algo="${algo##population_5_}"
algo="${algo%%_*}"
echo "algo="$algo

# play all against theirselves

for (( i = 0; i < 5; i++ )); do
    for (( j = 1; j < 11; j++ )); do
          file1="$folder_path/agent_id_"$i
          file2="$folder_path2/population_"$algo"_vs_heuristic_"$j"/agent_id_0"
          name=$folder_name"_test_vs_H_agent_id_"$i"_vs_trained_vs_h__"$j
          echo $file1
          echo $file2
          echo $name
          timeout 15m python3 src/main_popu_to_test.py \
          --config=popu_duo_"$algo"_test --env-config=sc2_compet \
          with 'env_args.map_name=3m_compet' \
          'env_args.log_more_stats=True' \
          'name='$name \
          'agent_type_1.checkpoint_path=["'$file1'", "'$file2'"]'

          sleep 10
          pkill python
          pkill Main_Thread
    done
done

deactivate