#!/usr/bin/env bash
#aze
# param1: method (maven/qmix/qvmix)
# param2: #_exp ([1-10])
# param3: #agent_id ([0-4])
# param4: adversaire (h, alone, vsh, qmix, maven, qvmix)
## param5: #_exp adversaire
## param6: #agent_id  if maven qmix qvmix
## example:
## maven 9 0 h
## maven 9 1 alone 10
## maven 9 4 vsh 7
algo=$1
folder="path_to_saved_models"
file1=$folder"/population_5_$1_$2/agent_id_"$3
case $4 in
"h")
  name="population_5_$1_$2_test_vs_h_agent_id_$3"
  python3 src/main_popu_to_test.py --config=popu_"$algo"_vs_heuristic_test --env-config=sc2_compet with env_args.map_name=3m_compet env_args.log_more_stats=True name="$name" agent_type_1.checkpoint_path=["'$file1'"]
  ;;

"alone")
  file2="$folder/population_alone_"$algo"_"$5"/agent_id_0"
  name="population_5_$1_$2_test_vs_A_agent_id_$3_vs_Alone_"$5
  python3 src/main_popu_to_test.py  --config=popu_duo_"$algo"_test --env-config=sc2_compet with env_args.map_name=3m_compet env_args.log_more_stats=True name="$name" agent_type_1.checkpoint_path=["'$file1'","'$file2'"]
;;

"vsh")
  file2="$folder/population_"$algo"_vs_heuristic_"$5"/agent_id_0"
  name="population_5_$1_$2_test_vs_H_agent_id_$3_vs_trained_vs_h__$5"
  python3 src/main_popu_to_test.py --config=popu_duo_"$algo"_test --env-config=sc2_compet with env_args.map_name=3m_compet env_args.log_more_stats=True name="$name" agent_type_1.checkpoint_path=["'$file1'","'$file2'"]
  ;;
"qmix"|"qvmix"|"maven")
  file2=$folder"/population_5_$4_$5/agent_id_"$6
  name="population_5_$1_$2_$3_vs_$4_$5_$6"
  python3 src/main_popu_to_test.py --config=popu_"$1"_vs_"$4"_test --env-config=sc2_compet with env_args.map_name=3m_compet env_args.log_more_stats=True name="$name" agent_type_1.checkpoint_path=["'$file1'", "'$file2'"]
  ;;
*)
  echo "args error"
  ;;
esac
