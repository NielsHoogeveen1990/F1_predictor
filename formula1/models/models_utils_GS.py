from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from formula1.preprocessing import get_clean_df
from formula1.models import RF
import pickle

def split_data(df):
    X = df.drop(columns='leftwon')
    y = df['leftwon']

    return train_test_split(X, y, random_state=42)


def fit(model, X_train, y_train):
    clf = model.pipeline()

    # clf.fit(X_train, y_train)

    gridsearch = GridSearchCV(clf, model.hyperparams(),
                              cv=5,
                              verbose=1,
                              refit=True,
                              scoring='accuracy')

    gridsearch.fit(X_train, y_train)

    # returns best model (with best parameters)
    return gridsearch.best_estimator_


def evaluate(y_hat, y_true):
    print('accuracy', accuracy_score(y_true, y_hat))
    print('f1_score', f1_score(y_true, y_hat, average='micro'))


def run():
    df = get_clean_df()

    X_train, X_test, y_train, y_test = split_data(df)

    fitted_model = fit(RF, X_train, y_train)

    y_hat = fitted_model.predict(X_test)

    evaluate(y_hat, y_test)

    with open('../formula1/model.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)

