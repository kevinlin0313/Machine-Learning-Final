from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math

class Preprocessor:
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def preprocess(self):
        # Apply various preprocessing methods on the DataFrame
        self.df.iloc[:, :17] = self._preprocess_numerical(self.df.iloc[:, :17])
        self.df.iloc[:, 17:] = self._preprocess_categorical(self.df.iloc[:, 17:])
        
        return self.df
        
    def _preprocess_numerical(self, df):
        # Custom logic for preprocessing numerical features goes here
        for column in df.columns: 
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column].fillna(mean_value,inplace=True)

        return df

    def _preprocess_categorical(self, df):
        # Add custom logic here for categorical features
        for column in df.columns: 
            if df[column].isnull().any():
                count_1 = df[column].eq(1).sum()
                count_0 = df[column].eq(0).sum()

                if count_1 > count_0: mode_value = 1 
                else : mode_value = 0
                
                df[column].fillna(mode_value,inplace=True)
        
        return df

    def _preprocess_ordinal(self, df):
        # Custom logic for preprocessing ordinal features goes here
        return df

# Implementing the classifiers (NaiveBayesClassifier, KNearestNeighbors, MultilayerPerceptron)

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        # Initialize the classifier
        self.class_probs = {}
        self.feature_stats = {}
        self.conditional_probabilities = {}
        self.proba_array = None
        pass

    def fit(self, X, y):
        # Implement the fitting logic for Naive Bayes classifier
        total_samples = len(y)
        class_counts = y.value_counts()
        for class_label, class_count in class_counts.items():
            self.class_probs[class_label] = class_count / total_samples
        
        for feature_idx in range(1, 18):  
            feature_key = f"F{feature_idx}"

            mean_f1_y1 = X.loc[y == 1, feature_key].mean()
            std_f1_y1 = X.loc[y == 1, feature_key].std()
            mean_f1_y0 = X.loc[y == 0, feature_key].mean()
            std_f1_y0 = X.loc[y == 0, feature_key].std()

            self.feature_stats[feature_key] = {
                'mean_y1': mean_f1_y1,
                'std_y1': std_f1_y1,
                'mean_y0': mean_f1_y0,
                'std_y0': std_f1_y0
            }
        
        for feature_idx in range(18, 78):  
            feature_key = f"F{feature_idx}"

            count_feature_1_given_y_0 = (X.loc[y == 0, feature_key] == 1).sum()
            count_feature_0_given_y_0 = (X.loc[y == 0, feature_key] == 0).sum()
            count_feature_1_given_y_1 = (X.loc[y == 1, feature_key] == 1).sum()
            count_feature_0_given_y_1 = (X.loc[y == 1, feature_key] == 0).sum()

            prob_feature_1_given_y_0 = count_feature_1_given_y_0 / (count_feature_1_given_y_0 + count_feature_0_given_y_0)
            prob_feature_0_given_y_0 = count_feature_0_given_y_0 / (count_feature_1_given_y_0 + count_feature_0_given_y_0)
            prob_feature_1_given_y_1 = count_feature_1_given_y_1 / (count_feature_1_given_y_1 + count_feature_0_given_y_1)
            prob_feature_0_given_y_1 = count_feature_0_given_y_1 / (count_feature_1_given_y_1 + count_feature_0_given_y_1)
            
            self.conditional_probabilities[feature_key] = {'10': prob_feature_1_given_y_0, '00': prob_feature_0_given_y_0, '11': prob_feature_1_given_y_1, '01': prob_feature_0_given_y_1}
        
        pass

    def predict(self, X):
        # Implement the prediction logic for Naive Bayes classifier
        predictions = []
        self.proba_array = np.zeros((X.shape[0], 2))
        i=0
        for index, row in X.iterrows():
            prob_y_0 = self.class_probs[0]
            prob_y_1 = self.class_probs[1]

            for feature_idx in range(1, 18):
                if feature_idx in [2, 10, 12, 14]:
                    continue
                
                feature_key = f"F{feature_idx}"
                feature_value = row[feature_key]

                if not pd.isna(feature_value):
                    if feature_idx == 10:
                        lambda0 = 1 / self.feature_stats[feature_key]['mean_y0']
                        prob_x_given_y_0 = lambda0 * math.exp(-lambda0 * feature_value)
                        
                        mean_y_1 = self.feature_stats[feature_key]['mean_y1']
                        std_y_1 = self.feature_stats[feature_key]['std_y1']
                        prob_x_given_y_1 = (1 / (math.sqrt(2 * math.pi) * std_y_1)) * math.exp(-(feature_value - mean_y_1)**2 / (2 * std_y_1**2))
                    else:
                        if feature_idx in [3, 11, 13, 15, 16, 17]:
                            lambda0 = 1 / self.feature_stats[feature_key]['mean_y0']
                            prob_x_given_y_0 = lambda0 * math.exp(-lambda0 * feature_value)
                            
                            lambda1 = 1 / self.feature_stats[feature_key]['mean_y1']
                            prob_x_given_y_1 = lambda1 * math.exp(-lambda1 * feature_value)
                        else:
                            mean_y_0 = self.feature_stats[feature_key]['mean_y0']
                            std_y_0 = self.feature_stats[feature_key]['std_y0']
                            mean_y_1 = self.feature_stats[feature_key]['mean_y1']
                            std_y_1 = self.feature_stats[feature_key]['std_y1']

                            prob_x_given_y_0 = (1 / (math.sqrt(2 * math.pi) * std_y_0)) * math.exp(-(feature_value - mean_y_0)**2 / (2 * std_y_0**2))
                            prob_x_given_y_1 = (1 / (math.sqrt(2 * math.pi) * std_y_1)) * math.exp(-(feature_value - mean_y_1)**2 / (2 * std_y_1**2))

                    prob_y_0 *= prob_x_given_y_0
                    prob_y_1 *= prob_x_given_y_1
            
            for feature_idx in range(18, 78):
                feature_key = f"F{feature_idx}"
                feature_value = row[feature_key]
                
                if not pd.isna(feature_value):
                    feature_value_int = int(feature_value)
                    prob_x_given_y_0 = self.conditional_probabilities[feature_key][str(feature_value_int) + '0']
                    prob_x_given_y_1 = self.conditional_probabilities[feature_key][str(feature_value_int) + '1']

                    prob_y_0 *= prob_x_given_y_0
                    prob_y_1 *= prob_x_given_y_1
        
            self.proba_array[i, 0] = prob_y_0
            self.proba_array[i, 1] = prob_y_1
            prediction = 0 if prob_y_0 > prob_y_1 else 1
            predictions.append(prediction)
            i=i+1
        return predictions
    
    
    def predict_proba(self, X):
        # Implement probability estimation for Naive Bayes classifier
        return self.proba_array

# K-Nearest Neighbors Classifier
class KNearestNeighbors(Classifier):
    def __init__(self, k=10):
        # Initialize KNN with k neighbors
        self.k = k
        self.X_train = None
        self.y_train = None
        self.proba_array = None

    def fit(self, X, y):
        # Store training data and labels for KNN
        subset_X = X.iloc[:, :17]
        subset_X2 = X.iloc[:, 17:]
        min_values = subset_X.min()
        max_values = subset_X.max()

        for col in subset_X.columns:
            subset_X[col] = (subset_X[col] - min_values[col]) / (max_values[col] - min_values[col])
        
        self.X_train = pd.concat([subset_X, subset_X2], axis=1)
        self.y_train = y
        pass

    def predict(self, X):
        # Implement the prediction logic for KNN
        self.proba_array = np.zeros((X.shape[0], 2))
        predictions=[]
        subset_X = X.iloc[:, :17]
        subset_X2 = X.iloc[:, 17:]
        min_values = subset_X.min()
        max_values = subset_X.max()

        for col in subset_X.columns:
            subset_X[col] = (subset_X[col] - min_values[col]) / (max_values[col] - min_values[col])
        
        X = pd.concat([subset_X, subset_X2], axis=1)
        j=0
        for index, row in X.iterrows():
            distances = []
            k_nearest_labels = []
            for index2, row2 in self.X_train.iterrows():
                distance = pow(np.sum((row.values - row2.values)**10),1/10)
                distances.append([distance,index2])
                
            distances.sort(key=lambda x: x[0])
            
            for i in range(self.k):
                label = self.y_train[distances[i][1]]
                k_nearest_labels.append(label)
                
            count_0 = 0
            count_1 = 0
            for i in range(len(k_nearest_labels)):
                if(k_nearest_labels[i]==0):
                    count_0 = count_0 + 1
                else:
                    count_1 = count_1 + 1
            if(count_0>count_1):
                prediction=0
            else:
                prediction=1
                
            predictions.append(prediction)
            self.proba_array[j][0] = count_0 / (count_0 + count_1)
            self.proba_array[j][1] = count_1 / (count_0 + count_1)
            j=j+1
        
        return predictions
    
    def predict_proba(self, X):
        # Implement probability estimation for KNN
        return self.proba_array

# Multilayer Perceptron Classifier
class MultilayerPerceptron(Classifier):
    def __init__(self):
        # Initialize MLP with given network structure
        self.input_size = 77
        self.hidden_layers_sizes = [2]
        self.output_size = 1
        self.weights = [
            np.random.randn(17, 1),
            np.random.randn(60, 1),    
            np.random.randn(2, 1)
        ]
        self.layer1_output = [0,0]
        self.proba_array = None
        pass

    def fit(self, X, y):
        # Implement training logic for MLP including forward and backward propagation
        mean = X.mean()
        std= X.std()
        X.iloc[:, :17] = (X.iloc[:, :17] - mean[:17]) / std[:17]
        epochs=100
        learning_rate=0.1
        for epoch in range(epochs):
            for index, row in X.iterrows():
                output = self._forward_propagation(row.values)
                self._backward_propagation_1(self.layer1_output, output, y[index], learning_rate)
                self._backward_propagation_2(row.values, self.layer1_output, output, y[index], learning_rate)
        pass

    def predict(self, X):
        # Implement prediction logic for MLP
        self.proba_array = np.zeros((X.shape[0], 2))
        predictions=[]
        mean = X.mean()
        std= X.std()
        X.iloc[:, :17] = (X.iloc[:, :17] - mean[:17]) / std[:17]
        i=0
        for index, row in X.iterrows():
            proba=self._forward_propagation(row.values)
            if proba >= 0.5 :
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)
            self.proba_array[i][0] = 1-proba
            self.proba_array[i][1] = proba
            i=i+1
         
        return predictions

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        print(self.proba_array)
        return self.proba_array
        
    def _forward_propagation(self, X):
        # Implement forward propagation for MLP
        layer1_net=[0,0]
        layer1_net[0] = np.dot(X[:17], self.weights[0])
        self.layer1_output[0] = self.sigmoid(layer1_net[0])

        layer1_net[1] = np.dot(X[17:], self.weights[1])
        self.layer1_output[1] = self.sigmoid(layer1_net[1])
        final_net = np.dot(self.weights[2].T,self.layer1_output)
        final_output = self.sigmoid(final_net)

        return final_output
    
    def _backward_propagation_1(self, input, output, target, learning_rate):
        # Implement backward propagation for MLP
        for i in range(len(self.weights[2])):
            arr = learning_rate * (target-output) * output * (1-output) * input[i]
            self.weights[2][i] += arr.reshape((1,))
        
    def _backward_propagation_2(self, input, output1, output2, target, learning_rate):
        # Implement backward propagation for MLP
        for i in range(len(self.weights[0])):
            arr = learning_rate * (target-output2) * output2 * (1-output2) * output1[0] * (1-output1[0]) * self.weights[1][0] * input[i]
            self.weights[0][i] += arr.reshape((1,))
        for i in range(len(self.weights[1])):
            arr = learning_rate * (target-output2) * output2 * (1-output2) * output1[1] * (1-output1[1]) * self.weights[1][1] * input[i+17]
            self.weights[1][i] += arr.reshape((1,))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    
# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    
    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, proba[:, 1])
        else:  # Multiclass classification
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
    
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('trainWithLabel.csv')
    
    # Preprocess the training data
    preprocessor = Preprocessor(df)
    df_processed = preprocessor.preprocess()

    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier(), 
              'KNN': KNearestNeighbors(), 
              'MLP': MultilayerPerceptron()
             
    }
    
    X_train = df_processed.drop('Outcome', axis=1)
    y_train = df_processed['Outcome']
    
    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []

    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)
            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)
    
    # Save results to an Excel file
    all_results_df.to_excel('cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")

    # Load the test dataset, assuming you have a test set CSV file without labels
    df_ = pd.read_csv('testWithoutLabel.csv')
    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")
    

if __name__ == "__main__":
    main()
