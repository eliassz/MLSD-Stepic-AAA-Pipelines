import argparse
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import mlflow


def main(model_name):
    # Define feature names
    features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    
    # Load the iris dataset
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns=features)
    iris_df['target'] = iris.target

    x = iris.data
    y = iris.target

    # Split the dataset
    test_size = 0.8
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=test_size, random_state=1
    )

    # Dictionary of models
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=1),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1),
        'LogisticRegression': LogisticRegression(random_state=1, max_iter=200)
    }

    # Check if the specified model is valid
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not a valid model. \
                          Choose from: {list(models.keys())}")

    # Select the model
    model = models[model_name]

    # Train the model
    model.fit(train_x, train_y)

    # Make predictions
    pred_test_y = model.predict(test_x)

    # Calculate metrics
    recal = recall_score(test_y, pred_test_y, average='weighted')
    prec = precision_score(test_y, pred_test_y, average='weighted')

    print('recall =', recal)
    print('prec =', prec)

    # Set up MLflow
    mlflow.set_tracking_uri('http://84.201.128.89:90/')
    mlflow.set_experiment('ivlomonosov-seminar-pipeline')

    with mlflow.start_run(run_name=model_name):
        mlflow.log_metrics({
            'prec': prec,
            'recal': recal,
        })
        mlflow.log_params({
            'model_name': model_name,
            'num_features': len(train_x[0]),
            'test_size': test_size,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the iris dataset \
            and log metrics with MLflow'
    )
    parser.add_argument(
        'model_name', type=str, help='The name of the model to train. \
                        Choose from: RandomForestClassifier, \
                        DecisionTreeClassifier, LogisticRegression')
    args = parser.parse_args()
    main(args.model_name)
