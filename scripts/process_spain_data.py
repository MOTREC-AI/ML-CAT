import pandas as pd
import numpy as np
from pathlib import Path
from functools import wraps
import datetime as dt
import re

DATADIR = Path("../assets/data_asset")
data = pd.read_excel(DATADIR / "final_data.xlsx", sheet_name="DATA")

col_map = {'patient_id': 'patient_id',
           'date_visit_0': 'date_visit_0',
           'vte_episode': 'vte_episode',
           'visit_when_vte': 'visit_when_vte',
           'date_cancer_dx': 'date_cancer_dx',
           'birth_date_(year)': 'birth_date_year',
           'ethnicity': 'ethnicity',
           'gender': 'SEX',
           'menopause_status': 'menopause_status',
           'weight_kg': 'weight_kg',
           'height_cm': 'height_cm',
           'body_surface_m2': 'body_surface_m2',
           'bmi_kg/m2': 'bmi_kg/m2',
           'bmi_category': 'bmi_category',
           'ecog': 'ecog',
           'hbp': 'hbp',
           'chf': 'chf',
           'dm': 'dm',
           'dlp': 'dlp',
           'cpd': 'cpd',
           'renal_insufficiency': 'renal_insufficiency',
           'hormonotherapy': 'hormonotherapy',
           'drug_hormono_id': 'drug_hormono_id',
           'reason_hormonot_id': 'reason_hormonot_id',
           'tabaquism': 'tabaquism',
           'majorsx_6_months_no_onco': 'majorsx_6_months_no_onco',
           'date_majorsx': 'date_majorsx',
           'majorsx_descrip': 'majorsx_descrip',
           'prior_vte_id': 'prior_vte_id',
           'prior_vte_1_location_id': 'prior_vte_1_location_id',
           'date_prior_vte_1_id': 'date_prior_vte_1_id',
           'prior_vte_2_location_id': 'prior_vte_2_location_id',
           'date_prior_vte_2_id': 'date_prior_vte_2_id',
           'prior_at_id': 'prior_at_id',
           'prior_at_descrip1_id': 'prior_at_descrip1_id',
           'prior_at1_if_other_descrip_id': 'prior_at1_if_other_descrip_id',
           'date_prior_at_1_id': 'date_prior_at_1_id',
           'prior_at_descrip_2_id': 'prior_at_descrip_2_id',
           'date_prior_at_2_id': 'date_prior_at_2_id',
           'prior_at_descrip_3_id': 'prior_at_descrip_3_id',
           'date_prior_at_3_id': 'date_prior_at_3_id',
           'vein_insuf_history_id': 'vein_insuf_history_id',
           'family_vte_history_id': 'family_vte_history_id',
           'tumoral_type_id': 'CANCER_TYPE_FINAL',
           'histology_id': 'histology_id',
           'tnm_id': 'tnm_id',
           'prior_onco_sx_id': 'prior_onco_sx_id',
           'date_prior_onco_sx': 'date_prior_onco_sx',
           'khorana_score': 'KS',
           'khorana_category': 'khorana_category',
           'permanent_catheter_id': 'permanent_catheter_id',
           'date_analytic': 'date_analytic',
           'albumin': 'ALBUMIN',
           'bilirrubin': 'TBILI',
           'creatinin': 'CREATININE',
           'alk_phos': 'ALKPHOS',
           'hemoglobin': 'HB',
           'leukocytes': 'WBC',
           'platelets': 'PLT',
           'inr': 'inr',
           'visit_vte': 'visit_vte',
           'vte_location1': 'vte_location1',
           'date_vte_1': 'date_vte_1',
           'vte_1_if_other_descrip': 'vte_1_if_other_descrip',
           'vte_type1': 'vte_type1',
           'vte_location2': 'vte_location2',
           'date_vte_2': 'date_vte_2',
           'vte_2_if_other_description': 'vte_2_if_other_description',
           '_vte_type2': '_vte_type2',
           'vte_location3': 'vte_location3',
           'date_vte_3': 'date_vte_3',
           'vte_type3': 'vte_type3',
           'at_event': 'at_event',
           'at_event_descrip': 'at_event_descrip',
           'date_at_event': 'date_at_event',
           'at_event_if_other': 'at_event_if_other',
           'patient_status_at_last_visit': 'patient_status_at_last_visit',
           'date_of_death': 'date_of_death',
           'date_for_last_visit_(if_death_that_date_also_qualifies_for_last_visit)': 'date_for_last_visit'}


def logg(f):
    @wraps(f)
    def wrapper(dataf, *args, **kwargs):
        tic = dt.datetime.now()
        result = f(dataf, *args, **kwargs)
        toc = dt.datetime.now()
        print(f"{f.__name__} took {toc - tic}\nInitial Shape: {dataf.shape}\nFinal Shape: {result.shape}")
        return result
    return wrapper


@logg
def start_pipeline(df):
    return df.copy()


@logg
def process_column_names(df):
    """
    Remove serial numbers and spaces from column names
    :param df: dataframe
    :return: df: dataframe with clean column names
    """
    cols = df.columns
    new_cols = {c: re.sub("^\d+\.\s?|^\d+a\.\s?", "", c).replace(" ", "_").lower() for c in cols}
    df.columns = list(new_cols.values())
    return df.rename(columns=col_map)


def set_event(dataf):
    events = np.where(dataf.date_vte_1.notna(), 1, np.where(dataf.date_of_death.notna(), 2, 0))
    print(f"Total vte: {(events==1).sum()}")
    print(f"Total deaths: {(events==2).sum()}")
    return events


@logg
def time_to_event(df):
    """
    Add time to event based columns in the dataset
    :param df: dataframe with clean column names
    :return: df: dataframe with survival columns
    """
    df["AGE"] = df.date_visit_0.dt.year - df.birth_date_year
    df["SEX"] = df.SEX.apply(lambda x: "F" if x == 1 else "M")  # 0:Male  1: Female
    df["DX_delta"] = (df.date_visit_0 - df.date_cancer_dx).dt.days
    # tnm stage IV is metastasis
    df["SAMPLE_TYPE"] = df.tnm_id.apply(lambda x: "Metastasis" if x ==  4 else "Local_Tumor")
    # CANCER_TYPE 2= colorrectal	3=esophagus	4=gastric	5=pancreatic	6=non small cell lung cancer
    cancer_type_map = {
        3: "esophagogastric",
        4: "esophagogastric",
        2: "colorectal",
        5: "pancreatic_adenocarcinoma",
        6: "lung"
    }
    df["CANCER_TYPE_FINAL"] = df.CANCER_TYPE_FINAL.map(cancer_type_map)
    # observed time from date of entry to study
    df["OBS_TIME_LAST_VISIT"] = (df.date_for_last_visit - df.date_visit_0).dt.days
    # observed time from first VTE event when the patient did not visit after VTE
    # df["OBS_TIME_VTE"] = np.where(df.visit_when_vte != 0,
    #                               (pd.to_datetime(df.date_vte_1, errors='ignore') - df.date_visit_0).dt.days, 0)
    df["OBS_TIME_VTE"] = (pd.to_datetime(df.date_vte_1) - df.date_visit_0).dt.days
    # observed time if patient died due to cancer
    df["OBS_TIME_DEATH"] = (pd.to_datetime(df.date_of_death, errors='ignore') - df.date_visit_0).dt.days
    # final observed time based on death/vte/censored
    df["OBS_TIME"] = np.where(df.date_vte_1.notna(), df.OBS_TIME_VTE,
                              np.where(df.date_of_death.notna(), df.OBS_TIME_DEATH, df.OBS_TIME_LAST_VISIT))
    df["HAD_CHEMO"] = True
    df["EVENT"] = set_event(df)
    # df = df[df.OBS_TIME > 0]
    return df


@logg
def test_dataset(df):
    # todo: recheck this logic
    # check all obs time > 0
    # assert df.OBS_TIME.min() > 0
    assert df.shape[0] == 391

    # check all vte event durations are correct
    assert ((pd.to_datetime(df[df.EVENT == 1].date_vte_1)
             - pd.to_datetime(df[df.EVENT == 1].date_visit_0)).dt.days
            - df[df.EVENT == 1].OBS_TIME).sum() == 0

    # check all death event durations are correct
    assert ((pd.to_datetime(df[df.EVENT == 2].date_of_death)
             - pd.to_datetime(df[df.EVENT == 2].date_visit_0)).dt.days
            - df[df.EVENT == 2].OBS_TIME).sum() == 0

    # check all death event durations are correct
    assert ((pd.to_datetime(df[df.EVENT == 0].date_for_last_visit)
             - pd.to_datetime(df[df.EVENT == 0].date_visit_0)).dt.days
            - df[df.EVENT == 0].OBS_TIME).sum() == 0
    return df


if __name__ == "__main__":
    a = (data
         .pipe(start_pipeline)
         .pipe(process_column_names)
         .pipe(time_to_event)
         .pipe(test_dataset))

    print(a.date_vte_1.notna().sum())
    print(a.EVENT.value_counts())
    a.to_csv(DATADIR / "spain_data_tte.csv", index=None)


