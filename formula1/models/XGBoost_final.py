from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from formula1.models.transformers import DTypeSelector
from formula1.models.transformers import DifferenceEncoder
from formula1.models.transformers import CorrFilterHighTotalCorrelation


def pipeline():

    numerical_pipeline = make_pipeline(
        DTypeSelector('number'),
        CorrFilterHighTotalCorrelation(),
        SimpleImputer(),
        StandardScaler()
    )

    object_pipeline = make_pipeline(
        DTypeSelector('object'),
        DifferenceEncoder(),
        #SimpleImputer(strategy='constant', fill_value='unkown')
        StandardScaler()
    )


    return make_pipeline(
        make_union(
            numerical_pipeline,
            object_pipeline,
        ),
        XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.1, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
    )


"""
This models_utils retrains with the entire dataset.

Optimal model:

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.1, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

"""
