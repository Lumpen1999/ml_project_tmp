
import numpy as np
import xgboost as xgb


class MyModel(Object):
    def __init__(self, task):
        self.task = task

class MyXGBoost(Object):
    def __init__(self, task):
        super().__init__()
        if task == 'class':
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()

    def fit(self, X_train, y_train):
        model.fit(X_train, y_train)

    def predict(self, X_test):
        y_hat = model.predict(X_test)
        return np.array(y_hat).astype(np.float64)

    def predict_proba(self, X_test):
        if self.task == 'class':
            y_hat_proba = model.predict_proba(X_test)
            y_hat_proba = np.array(y_hat_proba).astype(np.float64)
        else:
            y_hat_proba = None
        return y_hat_proba 

def get_model(model_name, task):
    if model_name == 'XGBoost':
        return MyXGBoost(task)
