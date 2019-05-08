from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from formula1.preprocessing import get_clean_df
from formula1.models import RF
from sklearn.model_selection import GroupKFold
import pickle

def group_split_data(df):
    X = df.drop(columns='leftwon')
    y = df['leftwon']

    group = df['year_left']
    group_kfold = GroupKFold(n_splits=5)
    train, test = next(group_kfold.split(X, y, group))

    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

    return X_train, X_test, y_train, y_test


def fit(model, X_train, y_train):
    clf = model.pipeline()

    # clf.fit(X_train, y_train)

    randomizedsearch = RandomizedSearchCV(clf, model.hyperparams(),
                              cv=GroupKFold(n_splits=5).split(X_train.drop(columns=['year_left']), y_train, X_train['year_left']),
                              n_iter= 3,
                              verbose=1,
                              refit=True,
                              scoring='accuracy')

    randomizedsearch.fit(X_train.drop(columns=['year_left']), y_train)

    # returns best model (with best parameters)
    return randomizedsearch.best_estimator_

def evaluate_CV(model, X_train, y_train):

    cv_score = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=GroupKFold(n_splits=5).split(X_train.drop(columns=['year_left']), y_train, X_train['year_left']),
        n_jobs=1,
        scoring='accuracy'
    )
    print('CV performance: %.3f +/- %.3f' % (np.mean(cv_score),np.std(cv_score)))


def evaluate_test(y_hat, y_true):
    print('accuracy', accuracy_score(y_true, y_hat))
    print('f1_score', f1_score(y_true, y_hat, average='micro'))


def run():
    df = get_clean_df()

    X_train, X_test, y_train, y_test = group_split_data(df)

    X_test = X_test.drop(columns=['year_left'])

    fitted_model = fit(RF, X_train, y_train)

    evaluate_CV(fitted_model, X_train, y_train)

    y_hat = fitted_model.predict(X_test)

    evaluate_test(y_hat, y_test)

    with open('../formula1/model2.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)

