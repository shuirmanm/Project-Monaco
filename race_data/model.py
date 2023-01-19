import pandas as pd
import sklearn

class Model:

    def __init__(self, model, eval_metric):
        self.model = model
        self.eval_metric = eval_metric

    def train(self, training_data):
        X_train = training_data.drop(['driver', 'podium'], axis = 1)
        y_train = training_data.podium
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

        self.model.fit(X_train, y_train)


    def test(self, testing_data):
        X_test = testing_data.drop(['driver', 'podium'], axis = 1)
        y_test = testing_data.podium
        
        y_pred = self.model.predict_proba(X_test)[:,:21]

        return self.eval_metric(y_test.to_numpy(), y_pred)

    def reset(self):
        self.model = sklearn.base.clone(self.model)