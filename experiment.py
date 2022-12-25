import numpy as np
import random
import warnings
import mlflow
import yaml
import hydra
from hydra.utils import get_original_cwd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder 

from utils import my_preprocessing, my_dataset, my_omegaconf
from utils.my_model import MyXGBoost
# import models

# RANDOM_STATE = random.choice(range(1, 100))
RANDOM_STATE = 44 
warnings.filterwarnings('ignore')

model_dict = {
    'XGBoost': MyXGBoost 
}


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    print('{}:{}'.format(cfg.experiment.model_name, cfg.experiment.dataset_name))

    # Original working directory
    owd = get_original_cwd()

    # load dataset info
    with open(f'{owd}/conf/all_datasets_info.yaml', 'r') as yml:
        all_datasets_info = yaml.safe_load(yml)
    dataset_info = all_datasets_info[cfg.experiment.dataset_name]

    # Set mflow 
    mlflow.set_tracking_uri('file://' + get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.experiment.dataset_name)

    eval_dict = {}
    with mlflow.start_run():

        my_omegaconf.log_params_from_omegaconf_dict(cfg)
        mlflow.log_params(dataset_info)

        for _ in range(cfg.experiment.n_folds):
            eval_dict = exe_fold(
                owd,
                cfg.experiment.model_name,
                dataset_info,
                cfg.experiment.dataset_name,
                cfg.experiment.n_splits,
                cfg.experiment.cat_encoding,
                eval_dict
                )

        for metrics_name, value_list in eval_dict.items():
            mean = np.mean(value_list)
            std =  np.std(value_list, ddof=1)
            mlflow.log_metric(key=f'{metrics_name}_mean', value=mean)
            mlflow.log_metric(key=f'{metrics_name}_std', value=std)
    return


def exe_fold(
    owd :str, 
    model_name: str,
    dataset_info: dict,
    dataset_name: str,
    n_splits: int,
    cat_encoding: str,
    eval_dict: dict
    ) -> dict:

    task = dataset_info['task']
    cat_col_index = dataset_info['cat_col_index']

    X, y = my_dataset.load_dataset(owd, dataset_name, dataset_info)
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)

    n_features = len(X.columns) 
    n_categorical = len(dataset_info['cat_col_index'])
    n_numerical = n_features - n_categorical

    mlflow.log_param(key='n_features', value=n_features)
    mlflow.log_param(key='n_categorical', value=n_categorical)
    mlflow.log_param(key='n_numerical', value=n_numerical)

    if task == 'class':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if cat_encoding == 'one_hot':
            X_train, X_test = my_preprocessing.one_hot_encoding(X_train, X_test, cat_col_index)
        elif cat_encoding == 'label':
            X_train, X_test = my_preprocessing.label_encoding(X_train, X_test, cat_col_index)

        if task == 'class':
            if dataset_info['label_encoding_for_target']:
                y_train, y_test = my_preprocessing.label_encoding_for_target(y_train, y_test)
        elif task == 'regression':
            X_train, X_test = my_preprocessing.standardization(X_train, X_test)
            if dataset_info['scaling_for_target']:
                y_train, y_test = my_preprocessing.standardization(y_train, y_test)
        elif task == 'multi':
            pass

        # to ndarray
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

        # predict
        model = my_model.get_model(model_name, task)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        y_hat_proba = model.predict_proba(X_test)

        # evaluate model
        if task == 'class':

            y_hat = np.clip(y_hat, 0, 1)

            # binary 
            acc = accuracy_score(y_test, y_hat)
            auc = roc_auc_score(y_test, y_hat)

            # prob 
            y_hat_proba = y_hat_proba.astype(np.float64)
            y_test = y_test.astype(np.float64)
            logloss = log_loss(y_test, y_hat_proba)

            mlflow.log_metric(key='auc', value=auc, step=i)
            mlflow.log_metric(key='acc', value=acc, step=i)
            mlflow.log_metric(key='logloss', value=logloss, step=i)

            # init metrics list
            if len(eval_dict.keys()) == 0:
                eval_dict['auc'] = []
                eval_dict['acc'] = []
                eval_dict['logloss'] = []

            eval_dict['auc'].append(auc)
            eval_dict['acc'].append(acc)
            eval_dict['logloss'].append(logloss)

        elif task == 'regression':
            mse = mean_squared_error(y_test, y_hat)
            mlflow.log_metric(key='MSE', value=mse, step=i)

            # init metrics list
            if len(eval_dict.keys()) == 0:
                eval_dict['mse'] = []

            eval_dict['mse'].append(mse)

        elif task == 'multi':
            pass

    return eval_dict


if __name__=='__main__':
    main()
