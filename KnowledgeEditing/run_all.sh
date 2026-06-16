#!/usr/bin/env bash
set -e
python baseline.py | tee outputs/task1_baseline.log
python edit_rome.py | tee outputs/task2_rome.log
python edit_memit.py | tee outputs/task3_memit.log
python evaluate.py | tee outputs/task4_evaluate.log
