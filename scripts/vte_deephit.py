import pandas as pd
import numpy as np

from sksurv.metrics import concordance_index_censored
import re
from urllib.parse import quote_plus
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from sklearn import metrics
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from collections import defaultdict
from utils import get_logger, VTEDataLoader
from pymongo import MongoClient

logger = get_logger("vte_deephit")


class CauseSpecificNet(torch.nn.Module):
    """Network structure same as in the DeepHit paper, INCLUDING residual connections."""

    def __init__(
            self,
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm=True,
            dropout=None,
    ):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes_shared[:-1],
            num_nodes_shared[-1],
            batch_norm,
            dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                in_features + num_nodes_shared[-1],
                num_nodes_indiv,
                out_features,
                batch_norm,
                dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)

        # Adding residual connections
        out = torch.cat((input, out), dim=1)

        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


class LabTransform(LabTransDiscreteTime):
    """
    Transform the labels
    """

    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype("int64")


def get_target(df):
    return df["times"].astype("float32"), df["event"].astype("float32")


def c_stat(preds, durations, events, duration_index, suffix):
    """
    Create several survival metrics
    
    Args:
    -----
      preds: predicts cumulative incidence function
      durations: Actual durations for data
      events: Actual events for data
      duration_index: model's duration index
      suffix: suffix to differentiate between metrics

    Returns:
    --------
       dict
    """

    # if not int_risk_ppv_thresh:
    #     int_risk_ppv_thresh = 0.089
    #     high_risk_ppv_thresh = 0.11
    time_of_interest = 180  # months - check labtransform in get_dataset

    cif1 = pd.DataFrame(preds[0], duration_index)
    cif2 = pd.DataFrame(preds[1], duration_index)
    ev1 = EvalSurv(1 - cif1, durations, events == 1, censor_surv="km")
    ev2 = EvalSurv(1 - cif2, durations, events == 2, censor_surv="km")

    events_vte = events == 1

    # passing CIF and not 1-CIF as the corcordance function needs estimated risk
    # of experiencing an event
    # Ref: https://scikit-survival.readthedocs.io/en/stable/api/generated 
    # /sksurv.metrics.concordance_index_censored.html
    vte_c_index_harrel = concordance_index_censored(events == 1,
                                                    durations, cif1.iloc[time_of_interest, :])[0]
    death_c_index_harrel = concordance_index_censored(
        events == 2, durations, cif2.iloc[time_of_interest, :]
    )[0]

    res = defaultdict(list)

    res["_".join(["td_c_idx_vte", suffix])] = [round(ev1.concordance_td("antolini"), 4)]
    res["_".join(["td_c_idx_death", suffix])] = [round(ev2.concordance_td("antolini"), 4)]
    res["_".join(["harrel_c_idx_vte", suffix])] = [round(vte_c_index_harrel, 4)]
    res["_".join(["harrel_c_idx_death", suffix])] = [round(death_c_index_harrel, 4)]
    res["_".join(["count", suffix])] = [len(events)]

    return res


def get_best_params(feature_set):
    """
    get best params from mongodb

    >>> get_best_params("basic_binarized")
    {'L2_par': 0.0701203753051926,
     'alpha_par': 0.28753605387110115,
     'd_indiv': 60.0,
     'd_shared': 44.0,
     'dropout': 0.45085994723637624,
     'eta_par': 0.8454411524041041,
     'lr': 0.06142414407438413,
     'patience_par': 4.0,
     'sigma_par': 0.1866882110304722,
     'w_indiv': 116.0,
     'w_shared': 64.0}
    """

    from ibm_watson_studio_lib import access_project_or_space

    wslib = access_project_or_space()
    vte_credentials = wslib.get_connection("vte")


    USERNAME = vte_credentials.get("username")
    PASSWORD = vte_credentials.get("password")
    HOST = vte_credentials.get("host")
    PORT = 27017
    DATABASE = vte_credentials.get("database")
    uri = "mongodb://%s:%s@%s:27017/?authSource=admin" % (quote_plus(USERNAME), quote_plus(PASSWORD), HOST)
    client = MongoClient(uri)
    db = client.get_database(DATABASE)
    pipeline = [
        {"$match": {"exp_key": feature_set}, },
        {"$group": {"_id": "$exp_key", "best_loss": {"$min": "$result.loss"}}},
    ]
    try:
        best_loss = list(db.jobs.aggregate(pipeline))[0]["best_loss"]
    except Exception as e:
        raise ValueError(
            f"No hyperparams present for this Feature Set: {feature_set}", e
        )

    
    best_params = list(
        db.jobs.find({"exp_key": feature_set, "result.loss": {"$eq": best_loss}})
    )[0]["misc"]["vals"]

    res = {}
    for key in best_params.keys():
        newkey = key.replace("x", "")
        if newkey == "lr":
            res[newkey] = best_params.get(key)[0]
            res[newkey] = round(res[newkey], 3)
        else:
            res[newkey] = round(best_params.get(key)[0], 2)
    print(f"Best Loss for Hyper Params: {best_loss}")
    print(res)
    return res


def get_datasets() -> dict:
    """
    Get data in a format required for DeepHit
    """
    dl = VTEDataLoader()
    x_train, x_test, y_train, y_train_6, y_test = dl.get_train_data()
    x_train_ks, x_test_ks, y_train_ks, y_train_6_ks, y_test_ks = dl.get_train_data(ks=1)

    # transform the target variables as expected by PyCox models
    num_durations = int(max(y_train["times"])) + 1  # for cut-points
    # num_durations = list(np.arange(0, float(y_train["times"].max()), 30))
    labtrans = LabTransform(num_durations)
    labtrans_6 = LabTransform(181)
    # labtrans_6 = LabTransform(list(np.arange(0, float(181), 30)))

    y_train = labtrans.fit_transform(*get_target(y_train))
    y_train_6 = labtrans_6.fit_transform(*get_target(y_train_6))
    y_test = labtrans_6.transform(*get_target(y_test))
    y_train_6_ks = labtrans_6.fit_transform(*get_target(y_train_6_ks))
    y_test_ks = labtrans_6.transform(*get_target(y_test_ks))

    datasets = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_train_6": y_train_6,
        "y_test": y_test,
        "x_train_ks": x_train_ks,
        "x_test_ks": x_test_ks,
        "y_train_ks": y_train_ks,
        "y_train_6_ks": y_train_6_ks,
        "y_test_ks": y_test_ks,
        "labtrans": labtrans,
        "labtrans_6": labtrans_6
    }
    return datasets