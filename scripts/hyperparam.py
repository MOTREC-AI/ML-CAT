"""Hyper parameter tuner
This script will select the best hyper-parameter for the
DeepHit model for a given feature set. The params would
be saved on a mongodb database.

The hyper-parameters are selected using a Bayesian optim-
zation technique which is implemented in python's HYPEROPT
library

Steps:
  1. define a hyper-parameter space
  2. For each feature
     2.1 Do a 5 fold CV for the ensemble model
     2.2 Return the c-index for each fold
  3. For optimization - optimize (1 - output of 2.2)
    3.1 Optimization will run for 100 evals
"""
import os
import sys
import numpy as np
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials
from sklearn.model_selection import StratifiedKFold


module_path = os.path.abspath(os.path.join("../scripts"))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import VTEDataLoader, get_feature_config, get_logger
from vte_deephit import LabTransform, get_target, c_stat
from run_models import train_deephit, get_preprocessed_datasets
import argparse

logger = get_logger("hyperparam")
feature_config = get_feature_config()



# Get Raw data
dl = VTEDataLoader()
x_train, x_test, y_train, y_train_6, y_test = dl.get_train_data()

# transform the target variables as expected by PyCox models
num_durations = int(max(y_train["times"])) + 1  # for cut-points
labtrans = LabTransform(num_durations)
labtrans_6 = LabTransform(181)

y_train = labtrans.fit_transform(*get_target(y_train))

from ibm_watson_studio_lib import access_project_or_space
wslib = access_project_or_space()
vte_credentials = wslib.get_connection("vte")

parameter_space = {
    "L2_par": hp.uniform("L2_par", 0.01, 0.1),
    "dropout": hp.uniform("dropout", 0.4, 0.7),
    "lr": hp.uniform("lr", 0.001, 0.1),
    "patience_par": hp.quniform("patience_par", 3, 6, 1),
    "w_shared": hp.quniform("w_shared", 32, 128, 4),
    "d_shared": hp.quniform("d_shared", 32, 128, 4),
    "w_indiv": hp.quniform("w_indiv", 32, 128, 4),
    "d_indiv": hp.quniform("d_indiv", 32, 128, 4),
    "eta_par": hp.uniform("eta_par", 0.5, 0.9),
    "alpha_par": hp.uniform("alpha_par", 0.1, 0.4),
    "sigma_par": hp.uniform("sigma_par", 0.05, 0.4),
}


def objective_hyperopt(kwargs):
    """Objective function to optimize

    Will Train 1 deephit model for each fold
    """
    cv_c_stat = []
    skf = StratifiedKFold(5)
    for train_idx, val_idx in skf.split(feature_train, y_train[1]):
        model = train_deephit(0,
                                feature_train[train_idx],
                                (y_train[0][train_idx],
                                y_train[1][train_idx]),
                                lab_trans=labtrans,
                                **kwargs)
        preds = model.predict_cif(feature_train[val_idx])
        en_c_stat = c_stat(preds,
                           y_train[0][val_idx],
                           y_train[1][val_idx],
                           model.duration_index,
                           suffix="ensemble_hyper")
        logger.info(f'c-index: {en_c_stat["td_c_idx_vte_ensemble_hyper"][0]}')
        cv_c_stat.append(en_c_stat["td_c_idx_vte_ensemble_hyper"][0])

    logger.info(f"Objective Function value: {1 - np.mean(cv_c_stat)}")
    return 1 - np.mean(cv_c_stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparam Tuning using HyperOpt')
    parser.add_argument("--feature", type=str, default=False, help="which feature set to run")
    args = parser.parse_args()

    feature = args.feature
    if feature not in feature_config["FEATURE_GROUPS"]:
        raise ValueError(f"{args.feature} is not a valid feature group")

    logger.info(f"Running hyperparam tuning for {feature}")
    (
        feature_train,
        feature_test
    ) = get_preprocessed_datasets(feature, x_train, x_test, None, None)

    print(vte_credentials)
    uri = f'mongo://{vte_credentials.get("username")}:{vte_credentials.get("password")}@{vte_credentials.get("host")}:27017/{vte_credentials.get("database")}/jobs?authSource=admin'
    print(uri)
    trials = MongoTrials(
        uri,
        exp_key=feature)

    print(
        "please run hyperopt-mongo-worker to start processing the hyperparams\n'hyperopt-mongo-worker --mongo=res_MIS_PelvicFractureT:wtim@tlvdbmg3.mskcc.org:27017/tmg_pelvicFracture/jobs?authSource=admin\&skip --poll-interval=0.1'")
    best = fmin(fn=objective_hyperopt,
                space=parameter_space,
                trials=trials,
                algo=tpe.suggest,
                max_evals=100)
    logger.info(f"**** Best Params ****\n{feature}\n{best}")
