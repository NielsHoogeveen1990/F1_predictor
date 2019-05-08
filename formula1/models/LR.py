from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from formula1.models.transformers import DTypeSelector
from formula1.models.transformers import DifferenceEncoder


def pipeline():

    numerical_pipeline = make_pipeline(
        DTypeSelector('number'),
        SimpleImputer(),
        StandardScaler(),

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
        LogisticRegression()
    )


def hyperparams():
    return {
    }
