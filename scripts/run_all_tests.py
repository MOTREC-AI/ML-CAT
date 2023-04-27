"""
This script helps run all the tests in parallel

"""
import multiprocessing
import os

from utils import get_feature_config

feature_config = get_feature_config()

test_runs = []

run_command = "CUDA_VISIBLE_DEVICES={0} python run_test.py --feature={1} --samples=200 --n 30 --device cuda --save True"

for i, feature in enumerate(feature_config["FEATURE_GROUPS"]):
    test_runs.append(run_command.format(i % 2, feature))


def execute(cmd):
    os.system(cmd)


process_pool = multiprocessing.Pool(processes=11)
process_pool.map(execute, test_runs)
