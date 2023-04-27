import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from joblib import load
from pycox.models import DeepHit
from sklearn.utils import resample

from utils import get_feature_config, get_logger, get_parent_dir
from vte_deephit import CauseSpecificNet, c_stat, get_datasets
from run_models import get_preprocessed_datasets

load_dotenv()

seed = int(os.getenv("RANDOM_SEED"))
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
feature_config = get_feature_config()


def get_test_results(feature, n=30, samples=200, persist=False, device="cpu"):
    datasets = get_datasets()

    x_train = datasets.get("x_train")
    x_test = datasets.get("x_test")
    x_train_ks = datasets.get("x_train_ks")
    x_test_ks = datasets.get("x_test_ks")
    y_test = datasets.get("y_test")
    y_test_ks = datasets.get("y_test_ks")

    # chemo test
    x_test_chemo = x_test.copy()
    x_test_chemo = x_test_chemo.reset_index()
    chemo_idx = x_test_chemo[x_test_chemo["HAD_CHEMO"]].index
    non_chemo_idx = x_test_chemo[x_test_chemo["HAD_CHEMO"]].index

    # only lung
    x_test_lung = x_test.copy()
    x_test_lung = x_test_chemo.reset_index()
    lung_idx = x_test_lung[x_test_lung.CANCER_TYPE_FINAL == 'lung'].index

    (
        feature_train,
        feature_test,
        feature_train_ks,
        feature_test_ks,
    ) = get_preprocessed_datasets(feature, x_train, x_test, x_train_ks, x_test_ks)

    logger.info(f"Running for feature: {feature}")
    params = load(get_parent_dir() / f"models/{feature}/params.pkl")

    models = []
    for i in range(n):
        net = CauseSpecificNet(**params)
        m = DeepHit(net, device=device)
        m.load_model_weights(get_parent_dir() / f"models/{feature}/model_{i}.pt")
        models.append(m)

    scores = []
    for j in range(samples):
        # full test
        sub_test, sub_test_y_0, sub_test_y_1 = resample(
            feature_test, y_test[0], y_test[1], stratify=y_test[1], random_state=j,
        )

        logger.debug(f"train shape: {sub_test.shape}")
        logger.debug(f"durations shape: {sub_test_y_0.shape}")
        logger.debug(f"durations values: {sub_test_y_0}")
        logger.debug(f"event shape: {sub_test_y_1.shape}")
        logger.debug(f"event values: {sub_test_y_1}")
        cifs = []

        # ks
        sub_test_ks, sub_test_y_0_ks, sub_test_y_1_ks = resample(
            feature_test_ks,
            y_test_ks[0],
            y_test_ks[1],
            stratify=y_test_ks[1],
            random_state=j,
        )
        cifs_ks = []

        # chemo
        sub_test_chemo, sub_test_y_0_chemo, sub_test_y_1_chemo = resample(
            feature_test[chemo_idx],
            y_test[0][chemo_idx],
            y_test[1][chemo_idx],
            stratify=y_test[1][chemo_idx],
            random_state=j,
        )
        cifs_chemo = []

        # non chemo
        sub_test_non_chemo, sub_test_y_0_non_chemo, sub_test_y_1_non_chemo = resample(
            feature_test[non_chemo_idx],
            y_test[0][non_chemo_idx],
            y_test[1][non_chemo_idx],
            stratify=y_test[1][non_chemo_idx],
            random_state=j,
        )
        cifs_non_chemo = []

        # patients with DX date less than a year
        sub_test_dx_365, sub_test_y_0_dx_365, sub_test_y_1_dx_365 = resample(
            feature_test[x_test["DX_delta"] <= 365],
            y_test[0][x_test["DX_delta"] <= 365],
            y_test[1][x_test["DX_delta"] <= 365],
            stratify=y_test[1][x_test["DX_delta"] <= 365],
            random_state=j,
        )
        cifs_dx_365 = []

        # patients with cancer type = lung
        sub_test_lung, sub_test_y_0_lung, sub_test_y_1_lung = resample(
            feature_test[lung_idx],
            y_test[0][lung_idx],
            y_test[1][lung_idx],
            stratify=y_test[1][lung_idx],
            random_state=j,
        )
        cifs_lung = []

        for sm in models:
            cifs.append(sm.predict_cif(sub_test))
            cifs_ks.append(sm.predict_cif(sub_test_ks))
            cifs_chemo.append(sm.predict_cif(sub_test_chemo))
            cifs_non_chemo.append(sm.predict_cif(sub_test_non_chemo))
            cifs_dx_365.append(sm.predict_cif(sub_test_dx_365))
            cifs_lung.append(sm.predict_cif(sub_test_lung))

        logger.debug(f"cifs length: {len(cifs)}")
        logger.debug(f"cifs[0] - shape - {cifs[0].shape}, values:{cifs[0]}")

        # get all CIF means
        cif = np.mean(cifs, dtype=np.float64, axis=0)
        cif_ks = np.mean(cifs_ks, dtype=np.float64, axis=0)
        cif_chemo = np.mean(cifs_chemo, dtype=np.float64, axis=0)
        cif_non_chemo = np.mean(cifs_non_chemo, dtype=np.float64, axis=0)
        cif_dx_365 = np.mean(cifs_dx_365, dtype=np.float64, axis=0)
        cif_lung = np.mean(cifs_lung, dtype=np.float64, axis=0)

        logger.debug(f"Final CIF shape: {cif.shape}")

        # update the values in dict
        c_stat_test = c_stat(
            cif,
            sub_test_y_0,
            sub_test_y_1,
            models[0].duration_index,
            suffix="bootstrap_test",
        )

        c_stat_test.update(
            c_stat(
                cif_ks,
                sub_test_y_0_ks,
                sub_test_y_1_ks,
                models[0].duration_index,
                suffix="bootstrap_test_ks",
            )
        )

        c_stat_test.update(
            c_stat(
                cif_chemo,
                sub_test_y_0_chemo,
                sub_test_y_1_chemo,
                models[0].duration_index,
                suffix="bootstrap_test_chemo",
            )
        )

        c_stat_test.update(
            c_stat(
                cif_non_chemo,
                sub_test_y_0_non_chemo,
                sub_test_y_1_non_chemo,
                models[0].duration_index,
                suffix="bootstrap_test_non_chemo",
            )
        )

        c_stat_test.update(
            c_stat(
                cif_dx_365,
                sub_test_y_0_dx_365,
                sub_test_y_1_dx_365,
                models[0].duration_index,
                suffix="bootstrap_test_dx_365",
            )
        )

        c_stat_test.update(
            c_stat(
                cif_lung,
                sub_test_y_0_lung,
                sub_test_y_1_lung,
                models[0].duration_index,
                suffix="bootstrap_test_lung",
            )
        )
        scores.append(c_stat_test)

    assert len(scores) == samples
    res = pd.concat([pd.DataFrame(df) for df in scores])
    res["feature"] = feature
    logger.info(
        f"C-index\n-----------\nVTE: {res.td_c_idx_vte_bootstrap_test.mean()}\nVTE(lung): {res.td_c_idx_vte_bootstrap_test_lung.mean()}")
    if persist:
        logger.info(f"saving results to: results/test_{feature}.csv")
        res.to_csv(get_parent_dir() / f"results/test_{feature}.csv", index=False)
    logger.info(f"Completed run for feature: {feature}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model")
    parser.add_argument("--save", default=False, help="save results")
    parser.add_argument("--feature", type=str, default=False, help="which feature set to run")
    parser.add_argument("--n", type=int, default=30, help="Number of models in the ensemble")
    parser.add_argument("--samples", type=int, default=200, help="samples in the bootstrap")
    parser.add_argument("--device", type=str, default="cpu", help="GPU to run on")

    args = parser.parse_args()
    device = torch.device(args.device)

    logger = get_logger("test_results")
    logger.info(f"Running on device: {device}")
    logger.info(f"torch: {torch.__version__}")

    if args.feature not in feature_config["FEATURE_GROUPS"]:
        raise ValueError(f"{args.feature} is not a valid feature group")

    logger.info(f"Running for feature Group {args.feature}")

    get_test_results(feature=args.feature, samples=args.samples, n=args.n, persist=args.save, device=device)
