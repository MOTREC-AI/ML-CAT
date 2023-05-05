"""Common Utilities"""
import os
import matplotlib
import matplotlib.pyplot as plt
from configparser import ConfigParser
from pathlib import Path
import lifelines
import seaborn as sns
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
import logging.config
import yaml
import scipy.stats as st
from sklearn.utils import resample
from typing import Union, List, Tuple

load_dotenv()


def get_parent_dir():
    """Return the root folder path for this project

    NOTE: this function might need change if the relative position of this file
    is changed. Also, if you copy this function to another file, make sure to check the
    relative file path OR run test cases
    """
    path = Path(__file__).parent.parent
    return path


with open(get_parent_dir() / 'configs/logging.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)


def get_logger(file_name):
    return logging.getLogger(file_name)


logger = get_logger(__name__)



def bootstrap_ci(data, func, true_col, pred_col, duration_col=None, n_bootstrap=200, alpha=0.05):
    """
    Calculate the confidence interval using bootstrapping.

    Parameters
    ----------
    data: DataFrame
        The data
        
    func: callable
        The function to compute the statistic of interest (e.g. np.mean, np.median)
    true_col: str
        The name of the True column values
    pred_col: str
        The name of the predicted probabilities
    n_bootstrap: int, optional
        The number of bootstrap samples to generate (default: 200)
    alpha: float, optional
        The desired significance level (default: 0.05)

    Returns
    -------
    tuple
        The lower, upper and mean of the confidence interval
    """
    stat = []
    for i in range(n_bootstrap):
        if not duration_col:
            y_true, y_pred = resample(data[true_col] == 1,
                                      data[pred_col],
                                      n_samples=len(data),
                                      random_state=i)
            stat.append(func(y_true, y_pred))
        else:
            y_true, y_pred, y_obs = resample(
                data[true_col] == 1,
                data[pred_col],
                data[duration_col],
                n_samples=len(data),
                random_state=i)
            stat.append(func(y_obs, -y_pred, y_true))
            
    lower = np.percentile(stat, 100 * (alpha / 2))
    upper = np.percentile(stat, 100 * (1 - alpha / 2))
    mean = np.mean(stat)
    return round(lower, 2), round(upper, 2), round(mean, 2), stat


def calc_ci(arr, ci=95):
    """This is a doctest for this function
    >>> calc_ci([1.1, 1.2, 1.3, 1, 1])
    (1.0057, 1.2343)
    """
    return f"({np.round(np.percentile(arr, (100 - ci) / 2), 2):.2f}, {np.round(np.percentile(arr, 100 - (100 - ci) / 2), 2):.2f})"


def get_feature_config():
    feature_config = ConfigParser()
    feature_config.read(get_parent_dir() / 'configs/feature_config.cfg')
    return feature_config


def plot_roc(df, ks_col, event_col, title=None, fname=None, save=False):
    """
    Args:
      df: dataframe with Khorona scores and events
      ks_col: Column with Khorana scores
      event_col: Column with VTE events
    """
    from sklearn import metrics
    ks_roc_data = df[[ks_col, event_col]].copy()
    ks_roc_data.loc[:, "VTE"] = ks_roc_data[event_col] == 1
    fpr, tpr, thresholds = metrics.roc_curve(
        ks_roc_data.VTE, ks_roc_data[ks_col], pos_label=1
    )
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(
        fpr,
        tpr,
        linestyle="--",
        marker="o",
        # color="darkorange",
        lw=2,
        label="ROC curve",
        clip_on=False,
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("%s, AUC = %.2f" % (title, auc))
    plt.legend(loc="lower right")

    if save:
        if not fname:
            raise ValueError("File name must be provided to save plot")
        if not title:
            raise UserWarning("Plot saved without title - consider adding one before saving")
        plt.savefig(get_parent_dir() / f"visualizations/{fname}.svg", dpi=300, format="svg")
    plt.show()


def get_estimated_cif(durations, events, event_of_interest=1, time_of_interest=180.0, calculate_variance=False):
    """Get Competing Risk Estimate """
    if isinstance(durations, pd.Series):
        durations = durations.values
    if isinstance(events, pd.Series):
        events = events.values
    ajf = lifelines.AalenJohansenFitter(calculate_variance=calculate_variance,
                                        jitter_level=0.00001,
                                        seed=int(os.environ["RANDOM_SEED"]))
    ajf.fit(durations, events, event_of_interest=event_of_interest)
    res = ajf.cumulative_density_.loc[time_of_interest].values[0]
    return res


def bootstrap_prevalence_vte(durations, events, n_bootstrap=200, time_of_interest=180.0):
    cifs = []
    for i in range(n_bootstrap):
        durations_sampled, events_sampled = resample(durations,
                                                     events,
                                                     n_samples=len(durations),
                                                     random_state=i)
        cifs.append(100*get_estimated_cif(durations_sampled,
                                          events_sampled,
                                          time_of_interest=time_of_interest))
    return cifs


def get_pair_counts_and_vte(df, ks_condition, cif_condition, alpha=0.05, time_of_interest=180.0):
    filtered_df = df[ks_condition & cif_condition]
    pair_count = len(filtered_df)
    vte_estimates = bootstrap_prevalence_vte(filtered_df["obs_time"],
                                             filtered_df["event"],
                                             n_bootstrap=2000,
                                             time_of_interest=time_of_interest)
    lower = np.percentile(vte_estimates, 100 * (alpha / 2))
    upper = np.percentile(vte_estimates, 100 * (1 - alpha / 2))
    mean = np.mean(vte_estimates)
    return pair_count, round(mean, 2), round(lower, 2), round(upper, 2)


def plot_grouped_risks(cif,
                       durations,
                       events,
                       time_of_interest=181,
                       event_of_interest=1,
                       q=5,
                       name=None,
                       save=False):
    """
    cif: cumulative risk of VTE and Death 
         Shape: (2, days, # patients)
    time_of_interest: Time point to define risk groups
    q: number of quantiles for the grouping
    """
    days_to_months = 30
    # get the risk of VTE at six months
    cif_180 = np.array(100*pd.Series(cif[0][time_of_interest]))
    # cuts_vte, bins = pd.qcut(cif_180, q=q, labels=[1, 2, 3, 4, 5], retbins=True)
    cuts_vte = np.where(cif_180 >= 9, 2, 1)
    print(pd.Series(cuts_vte).value_counts())
    unique_groups = np.unique(cuts_vte)
    fig, ax = plt.subplots(figsize=(16, 12))
    labels = ["Low Risk (<9%)", "High Risk (>=9%)"]

    for i, group in enumerate(unique_groups):
        ajf = lifelines.AalenJohansenFitter(
            calculate_variance=True,
            jitter_level=0.00001,
            seed=int(os.environ["RANDOM_SEED"]))
        ajf.fit(durations[cuts_vte == group],
                events[cuts_vte == group],
                event_of_interest=event_of_interest)
        # plots the estimate method = self._estimation_method = "cumulative_density_" for ajf
        ajf.plot(ax=ax, ci_show=True, label=f"Group {group}: {labels[i]}")
        # print(ajf.cumulative_density_)
    # Add a legend
    lgd = ax.legend(bbox_to_anchor=(0.2, 1), loc='upper center', ncol=1)
    tick_intervals = 30 if time_of_interest < 400 else 90
    tick_positions = np.arange(0, time_of_interest, tick_intervals)
    tick_labels = tick_positions // days_to_months
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Cumulative Incidence')
    ax.set_xlabel('Time (in Months)')
    plt.title(f"{name} (n={len(durations)})")

    if save:
        if not name:
            raise ValueError("Please provide filename")
        plt.savefig(
            get_parent_dir() / f"visualizations/grouped_risks_{name}.svg",
            dpi=300,
            format="svg",
            bbox_inches="tight",
            bbox_extra_artists=(lgd,)
        );


def plot_calibration(cif, events, durations, feature, bins=None, save=False, name=None):
    """
    cif: the vte risk at 6 months 
    name: filename
    events: event of interest
    durations: observaed toime
    """
    
    cif = np.array(100 * pd.Series(cif))
    if bins is None:
        cuts_vte, bins = pd.qcut(cif, q=5, labels=[1, 2, 3, 4, 5], retbins=True)
    else:
        labels = range(1, len(bins))
        cuts_vte = pd.cut(cif, labels=labels, bins=bins)

    prevalence_vte = {}
    vals = {}
    for i in range(len(bins)-1):
        k = i+1
        # get Aalen-Johansen Estimate of CIF as we have competing risks
        prevalence_vte[k] = bootstrap_prevalence_vte(durations=durations[cuts_vte == k], events=events[cuts_vte == k])
        vals[k] = cif[cuts_vte == k]
        
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(23, 15))

    sns.lineplot(
        x=vals.keys(),
        y=[np.array(a[1]).mean() for a in vals.items()],
        ax=axes,
        label="predicted",
    )

    
    dict_list = []
    for key, values_list in zip(prevalence_vte.keys(), prevalence_vte.values()):
        for value in values_list:
            dict_list.append({"group": key, "cif": value})

    df = pd.DataFrame(dict_list)

    sns.lineplot(
        x="group",
        y="cif",
        ax=axes,
        label="actual (95% interval)",
        # ci="sd",
        errorbar="pi",
        err_style='bars',
        data=df,
        color="k"
    )
    axes.set_title(f"VTE CIF by Risk Group - {feature} ({name})", pad=10)
    axes.set_ylim(0, 25)
    axes.set_xlabel(r"Risk Group")
    axes.set_ylabel(r"Cumulative Incidence (%)")
    axes.xaxis.set_tick_params(
        which="major", size=5, width=1, direction="in", top="on"
    )
    axes.yaxis.set_tick_params(
        which="major", size=10, width=2, direction="in", top="on"
    )
    axes.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    
    risk_text = ["Predicted Risk", "----------------------"]
    for i in range(len(bins) - 1):
        risk_text.append(f"Group {i+1}: {round(bins[i], 2)}% - {round(bins[i+1], 2)}%")
    axes.text(
        1,
        18,
        "\n".join(risk_text),
        fontfamily="sans-serif",
        fontsize="large",
        backgroundcolor="#D3D3D3"
    )

    if save:
        if not name:
            raise ValueError("Please provide filename")
        plt.savefig(
            get_parent_dir() / f"visualizations/calibration_{feature}_{name}.svg",
            dpi=300,
            format="svg",
        );

class VTEDataLoader:
    """This class is used to load data from the VTE dataset.

    Methods
    -------
    get_missing_columns
        Get the missing columns in the dataset

    get_raw_data
        Get the raw data from the dataset

    """

    def __init__(self):
        self.dir = get_parent_dir() / os.getenv("DATA_DIR")
        self.raw_data: pd.DataFrame = None
        self.raw_columns: list = None
        self.missing_columns: list = None
        self.data_dict: pd.DataFrame = None
        self.get_missing_columns()

    def get_missing_columns(self) -> None:
        """Sets missing_columns to a list of columns that are missing in the dataset"""
        self.get_raw_data()
        self.missing_columns = list(self.raw_data.isna().sum()[self.raw_data.isna().sum() > 0].index)

    def get_raw_data(self) -> pd.DataFrame:
        """Return the raw dataset without any processing

        :return: raw dataset
        """
        logger.info("Reading raw data ..")
        date_cols = ['TM_DX_DTE',
                     'PROCEDURE_DTE',
                     'REPORT_DTE',
                     'DATE_ADDED',
                     'CANCER_VTE_DATE',
                     'OLD_VTE_DATE',
                     'FIRST_NOTE_DATE',
                     'LAST_NOTE_DATE',
                     'DEATH_DTE']

        self.raw_data = pd.read_csv(self.dir / os.getenv("DATA_FILE"), parse_dates=date_cols)
        logger.info("Raw data shape: {}".format(self.raw_data.shape))
        logger.info("removing obs_time NA or 0")
        self.raw_data = self.raw_data[(self.raw_data['OBS_TIME'] > 0)]
        logger.info(self.raw_data.shape)
        return self.raw_data

    def get_clean_data(self):
        return self.__clean_data()

    def __clean_data(self, data: pd.DataFrame = None):
        """Clean the data by completing the steps below.

        1. Remove rows with no observation time
        2. CAP the CHEMO duration to a maximum of 28 days
        3. Only keep genes which have over 1.5% prevalence in the dataset

        Parameters
        ----------
        data: pd.DataFrame, optional
            The data to be cleaned. If None, the raw data will be used.

        Raises
        ------
        ValueError
            If dataset is empty
        """
        if data is not None:
            df = data.copy()
        else:
            df = self.raw_data.copy()

        if len(df) == 0:
            logger.error("Data is empty")
            raise ValueError("Data is empty")

        # only keep records where the patient was observed for at least a day
        df = df[df.OBS_TIME > 0]
        df.loc[df['EVENT'] == 3, 'EVENT'] = 1  # according to Dr Mantha collapse 3 to 1
        df.loc[df['EVENT_6'] == 3, 'EVENT_6'] = 1
        self.raw_columns = list(df.columns)

        # AT Dr Mantha, clip Chemo to 28 days
        logger.info("Clipping Max Chemo duration to 28 days ......")
        chemo_cols = list(df.filter(regex='CHEMO_').columns)
        logger.info("CHEMO columns: {}".format(chemo_cols))
        df.loc[:, chemo_cols] = df.loc[:, chemo_cols].clip(upper=28)

        # add a binary variable to indicate any chemo is last 28 days
        chemo_cols_non_ks = [col for col in df.columns if "CHEMO_" in col and "_ks" not in col]
        chemo_cols_ks = [col for col in df.columns if "CHEMO_" in col and "_ks" in col]
        df["HAD_CHEMO"] = np.any(df[chemo_cols_non_ks] < 28, axis=1).values
        df["HAD_CHEMO_ks"] = np.any(df[chemo_cols_ks] < 28, axis=1).values

        gene_data = df[list(df.filter(regex='_alt$').columns)]
        gene_cols_prevalence = gene_data.sum() / gene_data.shape[0]
        top_gene = df[list(gene_cols_prevalence[gene_cols_prevalence > 0.015].index)]

        logger.info(df.shape)
        logger.info("Dropping genes with less than 1.5% prevalence ......")
        df = df.drop(gene_data.columns, axis=1)
        logger.info(f"Dropped all gene columns {df.shape}")
        df1 = pd.concat([df, top_gene], axis=1)
        logger.info(f"Shape after adding top gene columns {df1.shape}")
        logger.info(f"Dropped {gene_data.shape[1] - top_gene.shape[1]} columns")
        logger.info(df1.head())
        logger.info(f'Selected gene columns - \n{top_gene.columns}')
        # self.raw_data = df1


        return df1

    def get_data_dictionary(self):
        """
        Return the data dictionary for columns in raw data

        :return: data dictionary dataframe
        """
        logger.info("Getting data dictionary ...")
        self.data_dict = pd.read_excel(self.dir / os.getenv("DATA_DICTIONARY_FILE"))
        return self.data_dict

    def get_train_data(self, data=None, ks=0):
        """Get the train and test data for the model

        Parameters
        ----------
        data: pd.DataFrame, optional
            The data to be cleaned. If None, the raw data will be used.
        competing: bool, optional
            If True, the target column will have 2 events (1 and 2)
        ks: int, optional
            If > 0, the features will be KS features ending in "_ks"

        Returns
        -------
        X_train: pd.DataFrame
            The train data
        X_test: pd.DataFrame
            The test data
        y_train: np.array
            The train target
        y_test: np.array
            The test target
        """

        if data is None:
            data = self.raw_data.copy()

        data = self.__clean_data(data)

        target_cols = {}
        if not ks:
            ks_cols = ["EVENT_6_ks", "OBS_TIME_ks", "OBS_TIME_6_ks"]
            data.drop(ks_cols, axis=1, inplace=True)
            target_cols = {"obs_time": "OBS_TIME",
                           "obs_time_6": "OBS_TIME_6",
                           "event": "EVENT",
                           "event_6": "EVENT_6"}
        elif ks:
            non_ks_cols = ["OBS_TIME", "OBS_TIME_6", "EVENT", "EVENT_6"]
            data.drop(non_ks_cols, axis=1, inplace=True)
            data = data[data.KS.notna() & (data.OBS_TIME_ks > 0)]
            # data = data[data.KS.notna()]
            data.OBS_TIME_ks.fillna(0, inplace=True)
            data.OBS_TIME_6_ks.fillna(0, inplace=True)
            target_cols = {"obs_time": "OBS_TIME_ks",
                           "obs_time_6": "OBS_TIME_6_ks",
                           "event": "EVENT_6_ks",
                           "event_6": "EVENT_6_ks"}

        # train, test = train_test_split(data, test_size=0.2, random_state=int(os.getenv('RANDOM_SEED')),
        #                                stratify=data[target_cols["event"]])

        train_seq = pd.read_csv(get_parent_dir() / "assets/data_asset/train_seq.csv")
        test_seq = pd.read_csv(get_parent_dir() / "assets/data_asset/test_seq.csv")

        train = data[data.AUDIT_SEQ.isin(train_seq.train)]
        test = data[data.AUDIT_SEQ.isin(test_seq.test)]

        features = [col for col in data.columns if col not in list(target_cols.values())]
        X_train, t_train, e_train, t_train_6, e_train_6 = train[features], \
                                                          train[target_cols["obs_time"]].values, \
                                                          train[target_cols["event"]].values, \
                                                          train[target_cols["obs_time_6"]].values, \
                                                          train[target_cols["event_6"]].values

        X_test, t_test, e_test = test[features], test[target_cols["obs_time_6"]].values, test[
            target_cols["event_6"]].values

        y_train = pd.DataFrame({"event": e_train, "times": t_train})
        y_train_6 = pd.DataFrame({"event": e_train_6, "times": t_train_6})
        y_test = pd.DataFrame({"event": e_test, "times": t_test})

        event_type = int

        y_train = np.array([tuple(a) for a in y_train.values],
                           dtype=list(zip(y_train.dtypes.index, [event_type, int])))

        y_train_6 = np.array([tuple(a) for a in y_train_6.values],
                             dtype=list(zip(y_train_6.dtypes.index, [event_type, int])))

        y_test = np.array([tuple(a) for a in y_test.values],
                          dtype=list(zip(y_test.dtypes.index, [event_type, int])))

        return X_train, X_test, y_train, y_train_6, y_test
