import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import itertools

def balance_dataset(file_path, arg):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Filter only POLIMERO PP and PE
    filtered_df = df[df['POLIMERO'].isin(['PP', 'PE'])]
    
    # Separate the data by POLIMERO
    pp_samples = filtered_df[filtered_df['POLIMERO'] == 'PP']
    pe_samples = filtered_df[filtered_df['POLIMERO'] == 'PE']
    
    pp_balanced = pp_samples.sample(n=531, random_state=arg) if len(pp_samples) >= 531 else pp_samples
    pe_balanced = pe_samples.sample(n=531, random_state=arg) if len(pe_samples) >= 531 else pe_samples
    
    # Combine the balanced samples
    balanced_df = pd.concat([pp_balanced, pe_balanced])
    
    return balanced_df

def process_data(file_path, arg):
    # Load the balanced dataset
    balanced_df = balance_dataset(file_path, arg)
    
    # Prepare the features and target variable
    X = balanced_df.drop(columns=['POLIMERO'])  # Drop target and ID
    y = balanced_df['POLIMERO']
    
    return X, y

def balance_classes(X, y, random_state):
    # Separate the two classes (assumes binary classification)
    X_class_0 = X[y == 'PE']
    y_class_0 = y[y == 'PE']
    
    X_class_1 = X[y == 'PP']
    y_class_1 = y[y == 'PP']
    
    # Split each class into 90% train, 10% test using random_state
    X_class_0_train, X_class_0_test, y_class_0_train, y_class_0_test = train_test_split(
        X_class_0, y_class_0, test_size=0.1, random_state=random_state)
    X_class_1_train, X_class_1_test, y_class_1_train, y_class_1_test = train_test_split(
        X_class_1, y_class_1, test_size=0.1, random_state=random_state)
    
    # Combine the train and test sets from both classes
    X_train = pd.concat([X_class_0_train, X_class_1_train])
    y_train = pd.concat([y_class_0_train, y_class_1_train])
    
    X_test = pd.concat([X_class_0_test, X_class_1_test])
    y_test = pd.concat([y_class_0_test, y_class_1_test])
    
    return X_train, X_test, y_train, y_test

def evaluate_models(file_path):
    results = []
    
    # Define the classifiers to use
    classifiers = {
        'SVM': SVC(random_state=32),
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(random_state=32),
        'Decision Tree': DecisionTreeClassifier(random_state=32),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=32)
    }

    best_results = []
    
    # Use fixed random states
    balance_random_state = 9
    model_random_state = 32
    
    # Load and balance the dataset with the fixed random state
    X, y = process_data(file_path, balance_random_state)

    # Balance classes and split into train and test
    X_train, X_test, y_train, y_test = balance_classes(X, y, balance_random_state)

    for classifier_name, model in classifiers.items():
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        
        # Storing the result for each classifier
        best_results.append({
            'Model': classifier_name,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Accuracy': accuracy,
            'Balance_Random_State': balance_random_state,
            'Model_Random_State': model_random_state
        })
    
    # Find the best result (highest f1-score)
    best_result = max(best_results, key=lambda x: x['F1_Score'])
    print(best_result)

    return best_results

def save_results_to_excel(results, output_file):
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)

# Main execution
file_path = 'pellet_ecc_senza_duplicati.xlsx'
output_file = 'ablations.xlsx'

best_results = evaluate_models(file_path)
save_results_to_excel(best_results, output_file)

print("Best results saved to Excel")
