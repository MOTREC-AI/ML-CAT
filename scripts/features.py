import ast
import os
import re

import numpy as np
from pathlib import Path
import joblib
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
# imputation
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import get_feature_config, get_logger, get_parent_dir

load_dotenv()

logger = get_logger(__name__)

feature_config = get_feature_config()
regex_cols = ast.literal_eval(feature_config['FEATURES']['REGEX_COLS'])

import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin



def get_regex_features(df, feature):
    """Get features using regex pattern. In this case, we are trying to remove
    KS features.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features.
    feature : str
        Feature which is configured to be regex.
    Returns
    -------
    list
        List of features.
    """
    data = df.copy()

    data = data[list(set(data.columns) - set(list(data.filter(regex=regex_cols["KS"]).columns.values)))]
    return list(data.filter(regex=regex_cols[feature]).columns.values)


def get_feature_list(df, feature_group, ks=0):
    """Get features from the feature_config.cfg file.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features.
    feature_group : str
        Feature group which is configured to be regex.
    ks : int
        KS value.
    Returns
    -------
    list
        List of features.
    """
    if feature_group not in feature_config['FEATURE_GROUPS']:
        raise ValueError(f'{feature_group} not in feature_config')

    feature_group = ast.literal_eval(feature_config['FEATURE_GROUPS'][feature_group])
    res = []
    ks_res = []
    for feature_set in feature_group:
        features = ast.literal_eval(feature_config['FEATURE_SETS'][feature_set])
        for feature in features:
            if feature in regex_cols:
                res += get_regex_features(df, feature)
            else:
                res.append(feature)

    for item in res:
        if f"{item}_ks" in df.columns:
            ks_res.append(f"{item}_ks")
        else:
            ks_res.append(item)

    logger.info(f"\nFeature Group: {feature_group}\nKS: {ks}\nFeatures:{res}\nKS Features:{ks_res}")
    if ks:
        return ks_res
    return res


def get_features_by_type(feature_list, feature_group):
    """
    Get features by type.
    Parameters
    ----------
    feature_list : list
        List of features.
    Returns
    -------
    list
        List of features by type
    Raises
    ------
    ValueError
        If feature type is not in feature_config
    """

    num_cols = []
    cat_cols = []
    auto_encode_cols = []

    feature_group = feature_group.upper()

    if feature_group in feature_config.sections() and 'NUMERICAL_COLS_REGEX' in feature_config[feature_group]:
        logger.info(f"Using model specific NUMERICAL_COLS_REGEX config for {feature_config}")
        num_cols_regex = "|".join(ast.literal_eval(feature_config[feature_group]['NUMERICAL_COLS_REGEX']))
    else:
        logger.info(f"Using generic NUMERICAL_COLS_REGEX config for {feature_config}")
        num_cols_regex = "|".join(ast.literal_eval(feature_config['FEATURES']['NUMERICAL_COLS_REGEX']))

    if feature_group in feature_config.sections() and 'CATEGORICAL_COLS_REGEX' in feature_config[feature_group]:
        logger.info(f"Using model specific CATEGORICAL_COLS_REGEX config for {feature_config}")
        cat_cols_regex = "|".join(ast.literal_eval(feature_config[feature_group]['CATEGORICAL_COLS_REGEX']))
    else:
        logger.info(f"Using generic CATEGORICAL_COLS_REGEX config for {feature_config}")
        cat_cols_regex = "|".join(ast.literal_eval(feature_config['FEATURES']['CATEGORICAL_COLS_REGEX']))

    if feature_group in feature_config.sections() and 'AUTO_ENCODE_COLS_REGEX' in feature_config[feature_group]:
        logger.info(f"Using model specific AUTO_ENCODE_COLS_REGEX config for {feature_config}")
        auto_encode_cols_regex = "|".join(ast.literal_eval(feature_config[feature_group]['AUTO_ENCODE_COLS_REGEX']))
    else:
        logger.info(f"Using generic AUTO_ENCODE_COLS_REGEX config for {feature_config}")
        auto_encode_cols_regex = "|".join(ast.literal_eval(feature_config['FEATURES']['AUTO_ENCODE_COLS_REGEX']))

    for item in feature_list:
        item = item.strip()
        if item in feature_config['FEATURES']['NUMERICAL_COLS'] or re.search(num_cols_regex, item) \
                or item.replace("_ks", "") in feature_config['FEATURES']['NUMERICAL_COLS']:
            num_cols += [item]
        elif item in feature_config['FEATURES']['CATEGORICAL_COLS'] or re.search(cat_cols_regex, item) \
                or item.replace("_ks", "") in feature_config['FEATURES']['CATEGORICAL_COLS']:
            cat_cols += [item]
        elif re.search(auto_encode_cols_regex, item):
            auto_encode_cols.append(item)
        else:
            raise ValueError(f'{item} not in feature_config')
        
    num_cols, cat_cols, auto_encode_cols = sorted(num_cols), sorted(cat_cols), sorted(auto_encode_cols)
    logger.info(f"\nNumerical Columns: {num_cols}\nCategorical Columns:"
                f"{cat_cols}\nAuto Encode Columns: {auto_encode_cols}")
    return num_cols, cat_cols, auto_encode_cols


def get_features(df, feature_group, ks=0):
    IMPUTE = ast.literal_eval(feature_config['THRESHOLD']['IMPUTE'])
    IMPUTE_TYPE = ast.literal_eval(feature_config['THRESHOLD']['IMPUTE_TYPE'])
    STANDARDIZE = ast.literal_eval(feature_config['THRESHOLD']['STANDARDIZE'])
    feature_list = get_feature_list(df, feature_group, ks)
    num_cols, cat_cols, auto_encode_cols = get_features_by_type(feature_list, feature_group)

    logger.info(f"\nImputing: {IMPUTE}\nImputing Type: {IMPUTE_TYPE}\nStandardizing: {STANDARDIZE}")
    transformers = []
    steps = []

    # # Define the pipeline
    # pipeline = Pipeline(steps=[('imputer', CategoricalImputer(cat_cols))])
    #
    # # Fit and transform the data
    # transformed_data = pipeline.fit_transform(df)
    #
    # # Print the transformed data
    # print(transformed_data)

    if IMPUTE:
        if IMPUTE_TYPE == 'iterative':
            steps.append(('imputer', IterativeImputer(max_iter=10, random_state=int(os.environ['RANDOM_SEED']))))
        elif IMPUTE_TYPE == 'median':
            steps.append(('imputer', SimpleImputer(strategy='median')))
        elif IMPUTE_TYPE == 'KNN':
            steps.append(('imputer', KNNImputer(n_neighbors=5, )))
        else:
            raise ValueError(f'{IMPUTE_TYPE} not in feature_config')

    if STANDARDIZE:
        steps.append(('scaler', StandardScaler()))

    # Transformers
    numeric_transformer = Pipeline(steps=steps)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='error', drop='first'))])
    binarizer = Binarizer()

    # pipeline
    transformers.append(('num', numeric_transformer, num_cols))
    transformers.append(('cat', categorical_transformer, cat_cols))
    transformers.append(('bin', binarizer, auto_encode_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers)

    pre_processing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    Path((get_parent_dir() / f"models/{feature_group}")).mkdir(parents=True, exist_ok=True)
    if ks == 0:
        joblib.dump(pre_processing_pipeline, get_parent_dir() / f"models/{feature_group}/preprocessing.joblib")
    return pre_processing_pipeline


def get_feature_out(estimator, feature_in):
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}'
                    for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    logger.info(f'\n*** Transformed Feature Names ***\n{output_features}')
    return output_features