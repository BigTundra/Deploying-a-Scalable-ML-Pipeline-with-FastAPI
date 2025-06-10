import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from ml.data import process_data
import joblib

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, use_gridsearch=False):
    """
    Trains an AdaBoost model (with optional hyperparameter tuning).
    
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    use_gridsearch : bool
        Whether to perform GridSearchCV for hyperparameter tuning.

    Returns
    -------
    model : AdaBoostClassifier
        Trained model.
    """
    if use_gridsearch:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'estimator': [
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2)
            ]
        }
        ada = AdaBoostClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=ada,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        print("Best F1 score:", grid_search.best_score_)
        model = grid_search.best_estimator_
    else:
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=100,
            learning_rate=1.0,
        )
        model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : AdaBoostClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model: 
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    joblib.dump(model, path)


def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    return joblib.load(path)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : AdaBoostClassifier
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    
    X_slice, y_slice, _, _ = process_data(
        data[data[column_name] == slice_value], 
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_slice) # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
