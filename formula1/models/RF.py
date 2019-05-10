from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from formula1.models.transformers import DTypeSelector
from formula1.models.transformers import DifferenceEncoder
from formula1.models.transformers import CorrFilterHighTotalCorrelation


def pipeline():

    numerical_pipeline = make_pipeline(
        DTypeSelector('number'),
        CorrFilterHighTotalCorrelation(),
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    object_pipeline = make_pipeline(
        DTypeSelector('object'),
        DifferenceEncoder(),
        #SimpleImputer(strategy='constant', fill_value='unkown')
    )


    return make_pipeline(
        make_union(
            numerical_pipeline,
            object_pipeline,
        ),
        RandomForestClassifier()
    )


def hyperparams():
    return {
        'randomforestclassifier__bootstrap': [True, False],
        #'randomforestclassifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        #'randomforestclassifier__max_features': ['auto', 'sqrt'],
        'randomforestclassifier__min_samples_split': [2, 5],
        'randomforestclassifier__n_estimators': [200, 400, 1000]
    }
