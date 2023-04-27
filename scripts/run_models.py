"""DeepHit Cross Validation Runner Script

This script allows the user to train DeepHit Model in parallel for
all defined feature groups in configs/feature_config.cfg

The script expects the following environment variables to be set:

DATA_DIR=assets/data_asset # raw data directory
DATA_DICTIONARY_FILE=data dictionary expanded cohort 3-21-2022.xlsx 
CONFIG_DIR=configs/ # all configs used in the code
LOG_DIR=logs/ # log location
DATA_FILE="prepped_core 3-04-2022 for ML team.csv" # patient data
DATA_DICT="data dictionary expanded cohort 3-21-2022.xlsx"
RANDOM_SEED=345945

Please create the directories are required and place the files in
expected locations

Methods:
--------
get_preprocessed_datasets: transforms the features to be fed to the final model
train_deephit: fit one model to the train set
train_ensemble_model: train the ensemble model
ensemble_c_stat: calculate the c_stat for the ensemble
run_model: run the ensemble models

"""
import os
import sys
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
import random
import torch
import torchtuples as tt
from dotenv import load_dotenv
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.utils import resample
from pycox.models import DeepHit

module_path = os.path.abspath(os.path.join("../scripts"))
if module_path not in sys.path:
    sys.path.append(module_path)

from features import get_features  # noqa
from utils import get_feature_config, get_logger, get_parent_dir  # noqa
from vte_deephit import (  # noqa
    CauseSpecificNet,
    c_stat,
    get_best_params,
    get_datasets,
    LabTransform
)

load_dotenv()

# get the logger
logger = get_logger("ensemble_model")


# get the data/config dir
data_dir = Path(os.getenv("DATA_DIR"))

seed = int(os.getenv("RANDOM_SEED"))
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def get_preprocessed_datasets(feature_set, train, test, train_ks=None, test_ks=None):
    """Feature transformer

    This function transforms the features to be fed to the final model

    Args:
    --------

    feature_set: the feature group for features
    x_train: the training set
    x_test: the test set
    x_train_ks: the train set for patients with Khorana Score
    x_test_ks: the test set for patients with Khorana Score


    Returns:
    --------

    tuple <np.ndarray>: transformed features
    """

    logger.info("Generating pre-processing pipeline")
    preprocessing = get_features(train, feature_set, ks=0)
    feature_train = preprocessing.fit_transform(train).astype("float32")
    feature_test = preprocessing.transform(test).astype("float32")

    # if save:
        # only save preprocessor when training
    Path((get_parent_dir() / f"models/{feature_set}")).mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing, get_parent_dir() / f"models/{feature_set}/preprocessing_fit.joblib")

    # handle sparse arrays
    if not isinstance(feature_train, np.ndarray):
        feature_train = feature_train.toarray()
    if not isinstance(feature_test, np.ndarray):
        feature_test = feature_test.toarray()

    if train_ks is not None:
        logger.info("Generating pre-processing pipeline for KS cohort")
        preprocessing_ks = get_features(train_ks, feature_set, ks=1)
        feature_train_ks = preprocessing_ks.fit_transform(train_ks).astype("float32")
        feature_test_ks = preprocessing_ks.transform(test_ks).astype("float32")
        if not isinstance(feature_train_ks, np.ndarray):
            feature_train_ks = feature_train_ks.toarray()
        if not isinstance(feature_test_ks, np.ndarray):
            feature_test_ks = feature_test_ks.toarray()
        return feature_train, feature_test, feature_train_ks, feature_test_ks
    else:
        return feature_train, feature_test


def train_deephit(i, train_x, train_y, device=None, **kwargs):
    """Fit one model to the train set

    Args:
    -------
        i: cross-validation run number
        feature_set: Feature set being trained on
        train_x: training set
        train_y: dependent for train set
        train_idx: training indices 

    Returns:
    --------

    model: the trained model

    """

    # since the data is imbalanced - passing the whole train set in a batch
    batch_size = train_x.shape[0]

    # get the best params from hyperparam selection

    num_nodes_shared = [int(kwargs.get("w_shared")), int(kwargs.get("d_shared"))]
    num_nodes_indiv = [int(kwargs.get("w_indiv")), int(kwargs.get("d_indiv"))]
    num_risks = int(kwargs.get("num_risks", train_y[1].max()))
    out_features = len(kwargs.get("lab_trans").cuts)
    batch_norm = bool(kwargs.get("batch_norm", True))
    dropout = float(kwargs.get("dropout"))
    learn_r = float(kwargs.get("lr"))
    in_features = train_x.shape[1]
    L2_par = float(kwargs.get("L2_par"))
    eta_par = float(kwargs.get("eta_par"))
    alpha_par = float(kwargs.get("alpha_par"))
    sigma_par = float(kwargs.get("sigma_par"))
    patience_par = float(kwargs.get("patience_par"))
    epochs = 30

    net = CauseSpecificNet(
        in_features,
        num_nodes_shared,
        num_nodes_indiv,
        num_risks,
        out_features,
        batch_norm,
        dropout,
    )

    callbacks = [
        tt.callbacks.EarlyStopping(
            metric="loss", dataset="train", patience=patience_par
        )
    ]
    optimizer = tt.optim.AdamWR(
        lr=learn_r, decoupled_weight_decay=L2_par, cycle_eta_multiplier=eta_par
    )

    model = DeepHit(
        net,
        optimizer,
        alpha=alpha_par,
        sigma=sigma_par,
        duration_index=kwargs.get("lab_trans").cuts,
        device=device
    )

    model.fit(
        train_x, train_y, batch_size, epochs, callbacks, verbose=True,
    )

    return model


def train_ensemble_model(f_train, durations, events, n, device=None, **kwargs):
    """Cross Validated Ensemble trainer

    This function does the following

    - Trains an ensemble model with n=10 models in each ensemble
    - Trains for 5 CVs with 4 repeats
    - Persists the results of all the runs to a spreadsheet in 
      the results/ directory

    Args:
    --------

    feature_set: the feature set to be trained
    f_train: training features
    f_train_ks: Khorana Scored training features
    n: number of models in each ensemble

    Returns:
    --------

    None

    """
    ensemble_results = []

    # cross validation splits

    models = []
    for j in range(n):  # number of DL models in ensemble
        logger.info(f"Running ensemble #{j}")
        # resample training data with replacement for each ensemble run
        sub_train_x, sub_train_y_0, sub_train_y_1 = resample(
            f_train,
            durations,
            events,
            replace=True,
            stratify=events,
            random_state=j,
        )
        sub_model = train_deephit(
            j, sub_train_x, (sub_train_y_0, sub_train_y_1), device=device, **kwargs
        )
        models.append(sub_model)
    return models


def ensemble_c_stat(ensemble_models, features, durations, events, suffix):
    """Ensemble C-stat

    Args
    --------
    ensemble_models:
    features:
    durations:
    events:
    suffix:

    Returns
    --------

    """
    cifs = []
    for sm in ensemble_models:
        cifs.append(sm.predict_cif(features))

    cif = np.mean(cifs, dtype=np.float32, axis=0)

    return c_stat(
        cif,
        durations,
        events,
        ensemble_models[0].duration_index,
        suffix=suffix,
    )


def run_model(feature, mode="cv", save=True, n=10, device=None, datasets={}, **kwargs):
    """A wrapper to encapsulate the run for a feature group
    """
    logger.info("Getting the data")

    x_train_local = datasets.get("x_train").copy()
    x_test_local = datasets.get("x_test").copy()
    x_train_ks_local = datasets.get("x_train_ks").copy()
    x_test_ks_local = datasets.get("x_test_ks").copy()

    logger.info(f"Creating model for Feature Set: {feature}")
    (
        feature_train,
        feature_test,
        feature_train_ks,
        feature_test_ks,
    ) = get_preprocessed_datasets(feature, x_train_local, x_test_local, x_train_ks_local, x_test_ks_local)

    try:
        best_params = get_best_params(feature)
    except ValueError as exp:
        logger.error(f"Skipping {feature} - do hyperparam tuning for this first\n{exp}")
        return

    # handle sparse arrays
    if not isinstance(feature_train, np.ndarray):
        feature_train = feature_train.toarray()
    if not isinstance(feature_test, np.ndarray):
        feature_test = feature_test.toarray()
    if not isinstance(feature_train_ks, np.ndarray):
        feature_train_ks = feature_train_ks.toarray()

    logger.info(f"best params for {feature}: {best_params}")

    y_train = datasets.get("y_train")
    y_train_6 = datasets.get("y_train_6")
    y_train_6_ks = datasets.get("y_train_6_ks")

    if mode == "cv":
        rskf = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=4, random_state=int(os.getenv("RANDOM_SEED"))
        )

        res_ensembles = []
        for i, (train_index, val_index) in enumerate(rskf.split(feature_train, y_train[1])):
            logger.info(f"Feature: {feature}\t CV: #{i}")
            # list of scores from each ensemble run
            val_scores_test = {}

            # for each train_index, train an ensemble model
            # Each ensemble model will train on slight variations of training data
            # but the class weight should be preserved
            train_x = feature_train[train_index]
            train_y = (y_train[0][train_index], y_train[1][train_index])
            val_x = feature_train[val_index]
            val_y = (
                y_train_6[0][val_index],
                y_train_6[1][val_index],
            )

            models = train_ensemble_model(train_x, train_y[0], train_y[1], n=n, device=device, **best_params, **kwargs)

            # get validation metric
            val_metrics = {}
            val_metrics.update(ensemble_c_stat(models, val_x, val_y[0], val_y[1], suffix="cv_val"))

            # get ks test metric
            ks_val_indices = pd.Index(val_index).intersection(x_train_ks_local.index)
            seq_test_ks = [
                i for i, j in enumerate(list(x_train_ks_local.index)) if j in ks_val_indices
            ]

            val_x_ks = feature_train_ks[seq_test_ks]
            val_y_ks = (y_train_6_ks[0][seq_test_ks], y_train_6_ks[1][seq_test_ks])
            val_metrics.update(ensemble_c_stat(models, val_x_ks, val_y_ks[0], val_y_ks[1], suffix="cv_val_ks"))

            # get ks train metric
            ks_train_indices = pd.Index(train_index).intersection(x_train_ks_local.index)
            seq_train_ks = [
                i for i, j in enumerate(list(x_train_ks_local.index)) if j in ks_train_indices
            ]

            val_x_train_ks = feature_train_ks[seq_train_ks]
            val_y_train_ks = (y_train_6_ks[0][seq_train_ks], y_train_6_ks[1][seq_train_ks])
            val_metrics.update(
                ensemble_c_stat(models, val_x_train_ks, val_y_train_ks[0], val_y_train_ks[1], suffix="cv_train_ks"))

            res_ensembles.append(val_metrics)
        res = pd.concat([pd.DataFrame(es) for es in res_ensembles])
        logger.info(res)

        if save:
            logger.info(f"Saving CV results for {feature} to results/cv_metrics_{feature}.csv")
            res.to_csv(get_parent_dir() / f"results/cv_metrics_{feature}.csv", index=False)
    else:
        # perform full training
        # create path to store models
        Path((get_parent_dir() / f"models/{feature}")).mkdir(parents=True, exist_ok=True)
        models = train_ensemble_model(feature_train, y_train[0], y_train[1], n=n, **best_params, **kwargs)
        if save:
            logger.info(f"Saving models for {feature} to models/{feature}")
            params = {
                "in_features": feature_train.shape[1],
                "num_nodes_shared": [int(best_params.get("w_shared")), int(best_params.get("d_shared"))],
                "num_nodes_indiv": [int(best_params.get("w_indiv")), int(best_params.get("d_indiv"))],
                "num_risks": int(y_train[1].max()),
                "out_features": len(labtrans.cuts),
                "batch_norm": True,
                "dropout": best_params.get("dropout"),
            }
            joblib.dump(params, get_parent_dir() / f"models/{feature}/params.pkl")
            for i, m in enumerate(models):
                m.save_model_weights(get_parent_dir() / f"models/{feature}/model_{i}.pt")
            logger.info(f"Saved models for feature {feature}")


if __name__ == "__main__":

    # get the command line params
    parser = argparse.ArgumentParser(description="Run the model")
    parser.add_argument("--mode", default="cv", help="cv or full")
    parser.add_argument("--save", default=True, help="save results")
    parser.add_argument("--feature", type=str, default=False, help="which feature set to run")
    parser.add_argument("--device", type=str, default="cpu", help="GPU to run on")
    parser.add_argument("--n", type=int, default=10, help="number of models in ensemble")
    args = parser.parse_args()

    feature_config = get_feature_config()
    if args.feature not in feature_config["FEATURE_GROUPS"]:
        raise ValueError(f"{args.feature} is not a valid feature group")

    device = torch.device(args.device)

    # Get Raw data
    logger.info(f"Running for feature Group {args.feature}")
    logger.info(f"Running on device: {device}")
    logger.info(f"torch: {torch.__version__}")

    datasets = get_datasets()
    labtrans = datasets.get("labtrans")

    for k, v in datasets.items():
        if isinstance(v, tuple):
            logger.info(f"{k}: {len(v[0])}")
        elif isinstance(v, LabTransform):
            logger.info(f"{k}: {v.cuts}")
        else:
            logger.info(f"{k}: {v.shape}")

    run_model(feature=args.feature,
              save=args.save,
              mode=args.mode,
              n=args.n,
              device=device,
              datasets=datasets,
              lab_trans=labtrans)
