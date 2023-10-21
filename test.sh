#!/bin/bash

set -euox pipefail

python3 decision_tree_test.py
python3 statistics_test.py
python3 loss_test.py
