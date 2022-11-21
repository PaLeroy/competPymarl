#!/usr/bin/env bash
source env/bin/activate

folder="path_to_models"
config="elo_qmix_no_h_test"
case $1 in
"self")
  files_qmix=()
for (( i=1; i<=10; i++ ))
do
  files_qmix+=("${folder}population_alone_qmix_$i/agent_id_0")
done
echo ${files_qmix[2]}
python3 src/main_popu_to_test.py --config="elo_qmix_no_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qmix_self_no_h_test" \
agent_type_1.checkpoint_path=["'${files_qmix[0]}', '${files_qmix[1]}', '${files_qmix[2]}', '${files_qmix[3]}', '${files_qmix[4]}', '${files_qmix[5]}', '${files_qmix[6]}', '${files_qmix[7]}', '${files_qmix[8]}', '${files_qmix[9]}'"]

  ;;
"vsh")
files_qmix=()
for (( i=1; i<=10; i++ ))
do
  files_qmix+=("${folder}population_qmix_vs_heuristic_$i/agent_id_0")
done
echo ${files_qmix[2]}
config="elo_qmix_vsh_no_h_test"
echo $config
python3 src/main_popu_to_test.py --config="elo_qmix_no_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qmix_vsh_no_h_test" \
agent_type_1.checkpoint_path=["'${files_qmix[0]}', '${files_qmix[1]}', '${files_qmix[2]}', '${files_qmix[3]}', '${files_qmix[4]}', '${files_qmix[5]}', '${files_qmix[6]}', '${files_qmix[7]}', '${files_qmix[8]}', '${files_qmix[9]}'"]
  ;;
"popu")

files_qmix=()
files_qmix_string=""
save_model_qmix_string=""
save_model_qmix_interval_string=""
load_step_qmix_string=""
for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_qmix_string+="'${folder}population_5_qmix_$i/agent_id_$j'"
    save_model_qmix_string+="False"
    save_model_qmix_interval_string+="20000"
    load_step_qmix_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_qmix_string+=","
        save_model_qmix_string+=","
        save_model_qmix_interval_string+=","
        load_step_qmix_string+=","
    fi
  done
done
echo ""
echo ""

echo $config
python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_qmix_popu_no_h_test" \
agent_type_1.number=50 \
agent_type_1.checkpoint_path=["$files_qmix_string"] \
agent_type_1.save_model_qmix=["$save_model_qmix_string"] \
agent_type_1.save_model_qmix_interval=["$save_model_qmix_interval_string"] \
agent_type_1.load_step_qmix=["$load_step_qmix_string"]
  ;;
"all")

files_qmix_string=""
save_model_qmix_string=""
save_model_qmix_interval_string=""
load_step_qmix_string=""
for (( i=1; i<=10; i++ ))
do
  files_qmix_string+="'${folder}population_alone_qmix_$i/agent_id_0'"
  save_model_qmix_string+="False"
  save_model_qmix_interval_string+="20000"
  load_step_qmix_string+="10000000"
  files_qmix_string+=","
        save_model_qmix_string+=","
        save_model_qmix_interval_string+=","
        load_step_qmix_string+=","
done

for (( i=1; i<=10; i++ ))
do
  files_qmix_string+="'${folder}population_qmix_vs_heuristic_$i/agent_id_0'"
  save_model_qmix_string+="False"
  save_model_qmix_interval_string+="20000"
  load_step_qmix_string+="10000000"
  files_qmix_string+=","
        save_model_qmix_string+=","
        save_model_qmix_interval_string+=","
        load_step_qmix_string+=","
done

for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_qmix_string+="'${folder}population_5_qmix_$i/agent_id_$j'"
    save_model_qmix_string+="False"
    save_model_qmix_interval_string+="20000"
    load_step_qmix_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_qmix_string+=","
        save_model_qmix_string+=","
        save_model_qmix_interval_string+=","
        load_step_qmix_string+=","
    fi
  done
  done
  echo $files_qmix_string
  echo $save_model_qmix_interval_string
  echo $save_model_qmix_string
  echo $load_step_qmix_string

python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
name="elo_qmix_all_no_h_test" \
agent_type_1.number=70 \
agent_type_1.checkpoint_path=["$files_qmix_string"] \
agent_type_1.save_model=["$save_model_qmix_string"] \
agent_type_1.save_model_interval=["$save_model_qmix_interval_string"] \
agent_type_1.load_step=["$load_step_qmix_string"]
;;
esac


