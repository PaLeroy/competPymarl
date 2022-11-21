#!/bin/bash

python3 -m venv env

source env/bin/activate

python3.6 -m pip install --upgrade pip
pip install -r requirements.txt

deactivate
