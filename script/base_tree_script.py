import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import mlflow

def main():
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
    
    # Model parameters
    model_name = 'RandomForestClassifier'
    model = RandomForestClassifier(random_state=1)
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
    
    with mlflow.start_run(run_name='tree'):
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
    main()
