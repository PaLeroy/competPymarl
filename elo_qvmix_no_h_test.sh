#!/usr/bin/env bash
source env/bin/activate

folder="path_to_models"
config="elo_qvmix_no_h_test"
case $1 in
"self")
  files=()
for (( i=1; i<=10; i++ ))
do
  files+=("${folder}population_alone_qvmix_$i/agent_id_0")
done
echo ${files[2]}
python3 src/main_popu_to_test.py --config="elo_qvmix_no_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qvmix_self_no_h_test" \
agent_type_1.checkpoint_path=["'${files[0]}', '${files[1]}', '${files[2]}', '${files[3]}', '${files[4]}', '${files[5]}', '${files[6]}', '${files[7]}', '${files[8]}', '${files[9]}'"]

  ;;
"vsh")
files=()
for (( i=1; i<=10; i++ ))
do
  files+=("${folder}population_qvmix_vs_heuristic_$i/agent_id_0")
done
echo ${files[2]}
config="elo_qvmix_vsh_no_h_test"
echo $config
python3 src/main_popu_to_test.py --config="elo_qvmix_no_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qvmix_vsh_no_h_test" \
agent_type_1.checkpoint_path=["'${files[0]}', '${files[1]}', '${files[2]}', '${files[3]}', '${files[4]}', '${files[5]}', '${files[6]}', '${files[7]}', '${files[8]}', '${files[9]}'"]
  ;;
"popu")

files=()
files_string_qvmix=""
save_model_qvmix_string=""
save_model_qvmix_interval_string=""
load_step_qvmix_string=""
for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_string_qvmix+="'${folder}population_5_qvmix_$i/agent_id_$j'"
    save_model_qvmix_string+="False"
    save_model_qvmix_interval_string+="20000"
    load_step_qvmix_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_string_qvmix+=","
        save_model_qvmix_string+=","
        save_model_qvmix_interval_string+=","
        load_step_qvmix_string+=","
    fi
  done
done
echo ""
echo ""

echo $config
python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qvmix_popu_no_h_test" \
agent_type_1.number=50 \
agent_type_1.checkpoint_path=["$files_string_qvmix"] \
agent_type_1.save_model=["$save_model_qvmix_string"] \
agent_type_1.save_model_interval=["$save_model_qvmix_interval_string"] \
agent_type_1.load_step=["$load_step_qvmix_string"]
  ;;
"all")

files_string_qvmix=""
save_model_qvmix_string=""
save_model_qvmix_interval_string=""
load_step_qvmix_string=""
for (( i=1; i<=10; i++ ))
do
  files_string_qvmix+="'${folder}population_alone_qvmix_$i/agent_id_0'"
  save_model_qvmix_string+="False"
  save_model_qvmix_interval_string+="20000"
  load_step_qvmix_string+="10000000"
  files_string_qvmix+=","
        save_model_qvmix_string+=","
        save_model_qvmix_interval_string+=","
        load_step_qvmix_string+=","
done

for (( i=1; i<=10; i++ ))
do
  files_string_qvmix+="'${folder}population_qvmix_vs_heuristic_$i/agent_id_0'"
  save_model_qvmix_string+="False"
  save_model_qvmix_interval_string+="20000"
  load_step_qvmix_string+="10000000"
  files_string_qvmix+=","
        save_model_qvmix_string+=","
        save_model_qvmix_interval_string+=","
        load_step_qvmix_string+=","
done

for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_string_qvmix+="'${folder}population_5_qvmix_$i/agent_id_$j'"
    save_model_qvmix_string+="False"
    save_model_qvmix_interval_string+="20000"
    load_step_qvmix_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_string_qvmix+=","
        save_model_qvmix_string+=","
        save_model_qvmix_interval_string+=","
        load_step_qvmix_string+=","
    fi
  done
  done
  echo $files_string_qvmix
  echo $save_model_qvmix_interval_string
  echo $save_model_qvmix_string
  echo $load_step_qvmix_string

python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
name="elo_qvmix_all_no_h_test" \
agent_type_1.number=70 \
agent_type_1.checkpoint_path=["$files_string_qvmix"] \
agent_type_1.save_model=["$save_model_qvmix_string"] \
agent_type_1.save_model_interval=["$save_model_qvmix_interval_string"] \
agent_type_1.load_step=["$load_step_qvmix_string"]
;;
esac


