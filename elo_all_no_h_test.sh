#!/usr/bin/env bash
source env/bin/activate

folder="path_to_models/"


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


python3 src/main_popu_to_test.py --config="elo_all_no_h_test" \
--env-config=sc2_compet with env_args.map_name=3m_compet \
name="elo_all_no_h_test" \
agent_type_1.number=70 \
agent_type_1.checkpoint_path=["$files_string_qvmix"] \
agent_type_1.save_model=["$save_model_qvmix_string"] \
agent_type_1.save_model_interval=["$save_model_qvmix_interval_string"] \
agent_type_1.load_step=["$load_step_qvmix_string"] \
agent_type_2.number=70 \
agent_type_2.checkpoint_path=["$files_string_maven"] \
agent_type_2.save_model=["$save_model_maven_string"] \
agent_type_2.save_model_interval=["$save_model_maven_interval_string"] \
agent_type_2.load_step=["$load_step_maven_string"] \
agent_type_3.number=70 \
agent_type_3.checkpoint_path=["$files_qmix_string"] \
agent_type_3.save_model=["$save_model_qmix_string"] \
agent_type_3.save_model_interval=["$save_model_qmix_interval_string"] \
agent_type_3.load_step=["$load_step_qmix_string"]
