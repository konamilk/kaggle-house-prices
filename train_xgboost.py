import pandas as pd
import numpy as np
from load_data import load_test_data, load_train_data
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

logger = getLogger(__name__)

DIR = 'result_tmp/'

if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train = load_train_data()
    df_test = load_test_data()

    x_train = df_train.drop(['SalePrice', 'Id'], axis=1)
    x_test = df_test.drop('Id', axis=1)
    y_train = df_train['SalePrice']

    x_all = x_train.append(x_test, ignore_index=True)

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    data_type = x_train.dtypes.reset_index(name='type')
    string_columns = data_type[(data_type.type != 'int64') & (data_type.type != 'float64')]

    logger.debug('label encoding for: {}'.format(string_columns))

    for c in string_columns['index']:
        le = LabelEncoder()
        le.fit(x_all[c].fillna('NA'))

        x_train[c] = le.transform(x_train[c].fillna('NA'))
        x_test[c] = le.transform(x_test[c].fillna('NA'))


    logger.info('data preparation end {}'.format(x_train.shape))

    logger.info('training begin {}'.format(x_train.shape))
    model = XGBRegressor(n_estimators=20, random_state=71)
    model.fit(x_train, y_train)

    logger.info('training end {}'.format(x_train.shape))

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    logger.info('end')
