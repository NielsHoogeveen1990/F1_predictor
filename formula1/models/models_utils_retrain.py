from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from formula1.models import XGBoost_final
from formula1.preprocessing_newdata import get_clean_df

"""
This models_utils retrains with the entire dataset.

Optimal model:

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.1, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
       
With 2018 data:    
       
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.4, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

"""

CURRENT_YEAR = 2019

def split_data(df):
    X = df.drop(columns='leftwon')
    y = df['leftwon']

    #return train_test_split(X, y, random_state=42)
    return X, y


def fit(model, X, y):
    clf = model.pipeline()
    clf.fit(X, y)

    return clf


# def evaluate(y_hat, y_true):
#     print('accuracy', accuracy_score(y_true, y_hat))
#     print('f1_score', f1_score(y_true, y_hat, average='micro'))


def run():
    df = get_clean_df()

    df = df[df['year_left'] != CURRENT_YEAR]

    df.drop('year_left', axis=1, inplace=True)

    X, y = split_data(df)

    fitted_model = fit(XGBoost_final, X, y)

    #y_hat = fitted_model.predict(X_test)

    #evaluate(y_hat, y_test)

    with open('trained_models/model.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)
