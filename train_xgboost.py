import matplotlib
import pandas as pd
import numpy as np
import pickle
from load_data import load_test_data, load_train_data
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, ParameterGrid

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

    handler = FileHandler(DIR + 'train_xgboost.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train = load_train_data()
    df_test = load_test_data()

    x_train = df_train.drop(['SalePrice', 'Id'], axis=1)
    x_test = df_test.drop('Id', axis=1)
    y_train = df_train['SalePrice'].values

    x_all = x_train.append(x_test, ignore_index=True)

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    data_type = x_train.dtypes.reset_index(name='type')
    string_columns = data_type[(data_type.type != 'int64') & (data_type.type != 'float64')]

    logger.debug('label encoding: {}'.format(string_columns))

    for c in string_columns['index']:
        le = LabelEncoder()
        le.fit(x_all[c].fillna('NA'))

        x_train[c] = le.transform(x_train[c].fillna('NA'))
        x_test[c] = le.transform(x_test[c].fillna('NA'))

    logger.info('data preparation end {}'.format(x_train.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    all_params = {'max_depth': [3],
                  'learning_rate': [0.1],
                  'min_child_weight': [0.3],
                  'n_estimators': [100],
                  'colsample_bytree': [0.8],
                  'colsample_bylevel': [0.8],
                  'reg_alpha': [0.1],
                  'seed': [0]
                  }

    min_rmse = 1000000000
    min_params = None

    logger.info('training begin {}'.format(x_train.shape))

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        list_gini_score = []
        list_rmse_score = []
        list_best_iterations = []

        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            trn_y = y_train[train_idx]

            val_x = x_train.iloc[valid_idx, :]
            val_y = y_train[valid_idx]

            model = XGBRegressor(**params)
            model.fit(trn_x,
                      trn_y,
                      eval_set=[(val_x, val_y)],
                      early_stopping_rounds=10000,
                      )

            pred = model.predict(val_x, ntree_limit=model.best_ntree_limit)

            sc_rmse = np.sqrt(np.mean((val_y - pred) ** 2))

            list_rmse_score.append(sc_rmse)
            list_best_iterations.append(model.best_iteration)

            logger.info('  RMSE: {}'.format(sc_rmse))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_rmse = np.mean(list_rmse_score)

        if min_rmse > sc_rmse:
            min_rmse = sc_rmse
            min_params = params

        logger.info('current min rmse:{}, params: {}'.format(min_rmse, min_params))

    logger.info('minimum params: {}'.format(min_params))
    logger.info('minimum rmse: {}'.format(min_rmse))

    model = XGBRegressor(**min_params)
    model.fit(x_train, y_train)

    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)

    logger.info('training end {}'.format(x_train.shape))

    with open(DIR + 'model.pkl', 'rb') as f:
        model = pickle.load(f)

    pred_test = model.predict(x_test)

    df_submit = pd.read_csv('./input/sample_submission.csv')
    df_submit['SalePrice'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)

    logger.info('end')
