#!/usr/bin/env bash
source env/bin/activate
python3 src/main.py --config=config_name --env-config=sc2_compet with env_args.map_name=3m_compet
deactivate
