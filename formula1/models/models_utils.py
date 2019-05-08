from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def split_data(df):
    X = df.drop(columns='leftwon')
    y = df['leftwon']

    return train_test_split(X, y, random_state=42)


def fit(model, X_train, y_train):
    clf = model.pipeline()
    clf.fit(X_train, y_train)

    return clf


def evaluate(y_hat, y_true):
    print('accuracy', accuracy_score(y_true, y_hat))
    print('f1_score', f1_score(y_true, y_hat, average='micro'))


def run(model, df):
    X_train, X_test, y_train, y_test = split_data(df)

    fitted_model = fit(model, X_train, y_train)

    y_hat = fitted_model.predict(X_test)

    evaluate(y_hat, y_test)
