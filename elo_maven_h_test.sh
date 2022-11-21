#!/usr/bin/env bash
source env/bin/activate

folder="path_to_models/"
config="elo_maven_h_test"
case $1 in
"self")
  files=()
for (( i=1; i<=10; i++ ))
do
  files+=("${folder}population_alone_maven_$i/agent_id_0")
done
echo ${files[2]}
python3 src/main_popu_to_test.py --config="elo_maven_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_maven_self_h_test" \
agent_type_1.checkpoint_path=["'${files[0]}', '${files[1]}', '${files[2]}', '${files[3]}', '${files[4]}', '${files[5]}', '${files[6]}', '${files[7]}', '${files[8]}', '${files[9]}'"]

  ;;
"vsh")
files=()
for (( i=1; i<=10; i++ ))
do
  files+=("${folder}population_maven_vs_heuristic_$i/agent_id_0")
done
echo ${files[2]}
config="elo_maven_vsh_h_test"
echo $config
python3 src/main_popu_to_test.py --config="elo_maven_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_maven_vsh_h_test" \
agent_type_1.checkpoint_path=["'${files[0]}', '${files[1]}', '${files[2]}', '${files[3]}', '${files[4]}', '${files[5]}', '${files[6]}', '${files[7]}', '${files[8]}', '${files[9]}'"]
  ;;
"popu")

files=()
files_string_maven=""
save_model_maven_string=""
save_model_maven_interval_string=""
load_step_maven_string=""
for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_string_maven+="'${folder}population_5_maven_$i/agent_id_$j'"
    save_model_maven_string+="False"
    save_model_maven_interval_string+="20000"
    load_step_maven_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_string_maven+=","
        save_model_maven_string+=","
        save_model_maven_interval_string+=","
        load_step_maven_string+=","
    fi
  done
done
echo ""
echo ""

echo $config
python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
env_args.log_more_stats=True \
name="elo_maven_popu_h_test" \
agent_type_1.number=50 \
agent_type_1.checkpoint_path=["$files_string_maven"] \
agent_type_1.save_model_maven=["$save_model_maven_string"] \
agent_type_1.save_model_maven_interval=["$save_model_maven_interval_string"] \
agent_type_1.load_step_maven=["$load_step_maven_string"]
  ;;
"all")

files_string_maven=""
save_model_maven_string=""
save_model_maven_interval_string=""
load_step_maven_string=""
for (( i=1; i<=10; i++ ))
do
  files_string_maven+="'${folder}population_alone_maven_$i/agent_id_0'"
  save_model_maven_string+="False"
  save_model_maven_interval_string+="20000"
  load_step_maven_string+="10000000"
  files_string_maven+=","
        save_model_maven_string+=","
        save_model_maven_interval_string+=","
        load_step_maven_string+=","
done

for (( i=1; i<=10; i++ ))
do
  files_string_maven+="'${folder}population_maven_vs_heuristic_$i/agent_id_0'"
  save_model_maven_string+="False"
  save_model_maven_interval_string+="20000"
  load_step_maven_string+="10000000"
  files_string_maven+=","
        save_model_maven_string+=","
        save_model_maven_interval_string+=","
        load_step_maven_string+=","
done

for (( i=1; i<=10; i++ ))
do
  for ((j=0; j<=4; j++))
  do
    files_string_maven+="'${folder}population_5_maven_$i/agent_id_$j'"
    save_model_maven_string+="False"
    save_model_maven_interval_string+="20000"
    load_step_maven_string+="10000000"
    if [[ $i -eq 10 ]]  && [[ $j -eq 4 ]]
    then
      :
    else
        files_string_maven+=","
        save_model_maven_string+=","
        save_model_maven_interval_string+=","
        load_step_maven_string+=","
    fi
  done
  done
  echo $files_string_maven
  echo $save_model_maven_interval_string
  echo $save_model_maven_string
  echo $load_step_maven_string

python3 src/main_popu_to_test.py --config="$config" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
name="elo_maven_all_h_test" \
agent_type_1.number=70 \
agent_type_1.checkpoint_path=["$files_string_maven"] \
agent_type_1.save_model_maven=["$save_model_maven_string"] \
agent_type_1.save_model_maven_interval=["$save_model_maven_interval_string"] \
agent_type_1.load_step_maven=["$load_step_maven_string"]
;;
esac