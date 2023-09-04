from layer import Layer
from mlp import Mlp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.datasets import fetch_openml


def iris():

    """
    Load and preprocess the Iris dataset.
    """

    data = load_iris()
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']
    labels = np.array([label_names[i] for i in labels])
    sc_X = StandardScaler()
    features=sc_X.fit_transform(features)

    return(features,labels,label_names)

def breast():

    """
    Load and preprocess the Breast Cancer dataset.
    """

    df_breast = pd.read_csv('breast_cancer_diagnostic.shuf.lrn.csv', sep=',')
    X = (df_breast.loc[:, df_breast.columns!='class'])
    y = (df_breast.iloc[:,df_breast.columns=='class'])
    label_names = np.unique(y)
    X = X.to_numpy()
    y = y.to_numpy()
    y = y.astype(int)
    sc_X = StandardScaler()
    X=sc_X.fit_transform(X)

    return(X,y,label_names)

def mice():

    """
    Load and preprocess the Mice Protein dataset.
    """
    data = fetch_openml(name='miceprotein', version=4, parser="auto")

    X = data.data 
    y = data.target  # Target variable
    
    X = np.array(X)
    col_means = np.nanmean(X, axis=0)
    na_indices = np.isnan(X)
    X[na_indices] = np.take(col_means, na_indices.nonzero()[1])
    y = np.array(y)
    label_names =  np.unique(y)

    return X,y, label_names

def custom_evaluate_model(y_true, y_pred, label_names):

    """
    Evaluate the Scores of the Custom MLP
    """
        
    if(isinstance(y_true[0], str)):
        y_pred = [label_names[i] for i in y_pred]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        max_length = max(len(y_true), len(y_pred))
        y_true = np.resize(y_true, max_length)
        y_pred = np.resize(y_pred, max_length)
    labels= label_names
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in labels and pred_label in labels:
            true_idx = np.where(labels == true_label)[0][0]
            pred_idx = np.where(labels == pred_label)[0][0]
            confusion_matrix[true_idx, pred_idx] += 1
    
    epsilon = 1e-7  # small value to avoid division by zero
    accuracy = np.trace(confusion_matrix) / (np.sum(confusion_matrix) + epsilon)
    denominator = np.sum(confusion_matrix, axis=0)
    precision = np.where(denominator != 0, np.diagonal(confusion_matrix) / (denominator + epsilon), 0)
    recall = np.where(np.sum(confusion_matrix, axis=1) != 0, np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + epsilon), 0)
    f1 = np.where((precision + recall) != 0, 2 * (precision * recall) / (precision + recall + epsilon), 0)

    
    return accuracy, precision, recall, f1

def evaluate_model(y_true, y_pred, label_names):

    """
    Evaluate the Scores of the scikit algorithms
    """
        
    y_true = np.resize(y_true, len(y_pred))
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    
    y_true_encoded = np.array([label_mapping[label] for label in y_true])
    y_pred_encoded = np.array([label_mapping[label] for label in y_pred])
    
    if y_true_encoded.shape != y_pred_encoded.shape:
        max_length = max(len(y_true_encoded), len(y_pred_encoded))
        y_true_encoded = np.resize(y_true_encoded, max_length)
        y_pred_encoded = np.resize(y_pred_encoded, max_length)
    
    labels = np.array(list(label_mapping.keys()))
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in labels and pred_label in labels:
            true_idx = np.where(labels == true_label)[0][0]
            pred_idx = np.where(labels == pred_label)[0][0]
            confusion_matrix[true_idx, pred_idx] += 1
    
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    denominator = np.sum(confusion_matrix, axis=0)
    precision = np.where(denominator != 0, np.diagonal(confusion_matrix) / denominator, 0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    recall[np.isnan(recall)] = 0 
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0 
    return accuracy, precision, recall, f1

def cross_validation(X, y, layers_number,layer_size, activation_function, alpha, label_names):

    """
    Cross Validation
    """

    k = 5
    n_samples = X.shape[0]
    fold_size = n_samples // k

    indices = np.random.permutation(n_samples)

    scores = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(k):
        test_indices = indices[i*fold_size : (i+1)*fold_size]
        train_indices = np.concatenate((indices[:i*fold_size], indices[(i+1)*fold_size:]))

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        """
        Uncomment the following lines for the custom MLP
        """
        mlp = Mlp(X,layers_number, layer_size, activation_function, len(label_names), alpha, classes=label_names)
        mlp.fit(X_train, y_train,5000)
        y_pred = mlp.predict(X_test) 
        accuracy, precision, recall, f1_score = custom_evaluate_model(y_test, y_pred, label_names)
        """
        Uncomment the following lines for the scikit MLP
        """
        #SCIKITMLP = MLPClassifier()
        #SCIKITMLP.fit(X_train, y_train)
        #y_pred = SCIKITMLP.predict(X_test)
        #accuracy, precision, recall, f1_score = evaluate_model(y_test, y_pred, label_names)
        """
        Uncomment the following lines for the scikit DT
        """
        #dt_classifier = DecisionTreeClassifier()
        #dt_classifier.fit(X_train, y_train)
        #y_pred = dt_classifier.predict(X_test)      
        #accuracy, precision, recall, f1_score = evaluate_model(y_test, y_pred, label_names)

        scores.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    score = np.mean(scores)
    precision = np.mean(np.asarray(precisions, dtype=float))
    recall = np.mean(recalls)
    f1_score = np.mean(f1_scores)

    return score, precision, recall, f1_score
    

def mlp_grid_search(X, y, hidden_layers, hidden_layer_sizes, activation_functions, num_iterations, alphas, label_names):
    best_score = None
    best_params = {}

    for num_layers in hidden_layers:
        for layer_size in hidden_layer_sizes:
            for alpha in alphas:
                for activation_function in activation_functions:
                    scores = []
                    for _ in range(num_iterations):
                        
                        scores = cross_validation(X, y, num_layers,layer_size, activation_function, alpha, label_names)

                    mean_score = np.mean(scores)
                    print(mean_score)

                    if best_score is None or mean_score > best_score:
                        print("new best")
                        best_score = mean_score
                        best_params = {
                            'hidden_layers': num_layers,
                            'hidden_layer_size': layer_size,
                            'alphas': alpha,
                            'activation_function': activation_function,
                            'score': best_score
                        }
    return best_params

def write_parameters_to_file(parameters, dataset_name):
    try:
        with open('/Users/niklaspriewasser/Documents/TU/Machine Learning/Exc2/mlp - Copy/parameters.txt', 'a') as file:
            file.write(f"Dataset: {dataset_name}\n")
            file.write("Parameters:\n")
            for key, value in parameters.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            print("written")
            file.close()
    except Exception as e:
        print(f"Error occurred while writing parameters: {str(e)}")


"""
Uncomment the following lines for the Gridsearch
"""

#hidden_layers = [1, 2, 3, 4, 9, 10, 20]
#hidden_layer_sizes = [10, 20, 50, 60, 100] 
#alphas = [0.001,0.05,0.0001]
#activation_functions = ["sigmoid", "tanh", "relu",""]
#num_iterations = 1


#for i in range(0,3):
#    if(i == 0):
#        X, y,label_names = iris()
#        best_params_iris = mlp_grid_search(X, y, hidden_layers, hidden_layer_sizes, activation_functions, num_iterations, alphas, label_names)
#        write_parameters_to_file(best_params_iris, "iris")
#    if(i == 1):
#        X, y,label_names = breast()
#        best_params_breast = mlp_grid_search(X, y, hidden_layers, hidden_layer_sizes, activation_functions, num_iterations, alphas, label_names)
#        write_parameters_to_file(best_params_breast, "breast")
#    if(i == 2):
#        X, y,label_names = mice()
#        best_params_arr = mlp_grid_search(X, y, hidden_layers, hidden_layer_sizes, activation_functions, num_iterations, alphas, label_names)
#        write_parameters_to_file(best_params_arr, "mice")


"""
The following lines are used to run the chosen algo with the chosen dataset using cross validation 
"""

start_time = time.time()

"""
Choose dataset: mice, iris, breast
"""

X, y,label_names = mice()


"""
Hyperparameters
"""

hidden_layers= 1
hidden_layer_size= 10
alphas= 0.001
activation_function= ''

accuracy, precision, recall, f1_score = cross_validation(X, y,hidden_layers,hidden_layer_size, activation_function, alphas, label_names)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)
runtime = time.time() - start_time
print("Runtime:", runtime, "seconds")