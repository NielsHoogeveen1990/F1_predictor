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
        XGBClassifier()
    )


def hyperparams():
    return {
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__max_depth': [3,10],
        'xgbclassifier__min_child_weight': [1,5],
        'xgbclassifier__gamma': [0.1,0.4]
    }
