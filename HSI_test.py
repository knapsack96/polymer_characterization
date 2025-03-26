import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def balance_dataset(file_path, arg):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Filter only POLIMERO PP and PE
    filtered_df = df[df['POLIMERO'].isin(['PP', 'PE'])]
    
    # Separate the data by POLIMERO
    pp_samples = filtered_df[filtered_df['POLIMERO'] == 'PP']
    pe_samples = filtered_df[filtered_df['POLIMERO'] == 'PE']
    
    print(f"Original counts: PP={len(pp_samples)}, PE={len(pe_samples)}")
    
    pp_balanced = pp_samples.sample(n=531, random_state=arg) if len(pp_samples) >= 531 else pp_samples
    pe_balanced = pe_samples.sample(n=531, random_state=arg) if len(pe_samples) >= 531 else pe_samples
    
    print(f"Balanced counts: PP={len(pp_balanced)}, PE={len(pe_balanced)}")
    
    # Combine the balanced samples
    balanced_df = pd.concat([pp_balanced, pe_balanced])
    
    return balanced_df

def balance_classes(X, y, random_state):
    # Separate the two classes
    X_class_0 = X[y == 'PE']
    y_class_0 = y[y == 'PE']
    
    X_class_1 = X[y == 'PP']
    y_class_1 = y[y == 'PP']
    
    # Split each class into 90% train, 10% validation
    X_class_0_train, X_class_0_val, y_class_0_train, y_class_0_val = train_test_split(
        X_class_0, y_class_0, test_size=0.1, random_state=random_state)
    X_class_1_train, X_class_1_val, y_class_1_train, y_class_1_val = train_test_split(
        X_class_1, y_class_1, test_size=0.1, random_state=random_state)
    
    # Combine the train and validation sets
    X_train = pd.concat([X_class_0_train, X_class_1_train])
    y_train = pd.concat([y_class_0_train, y_class_1_train])
    
    X_val = pd.concat([X_class_0_val, X_class_1_val])
    y_val = pd.concat([y_class_0_val, y_class_1_val])
    
    print(f"Training set size: {len(y_train)}, Validation set size: {len(y_val)}")
    print("Training set class distribution:\n", y_train.value_counts())
    print("Validation set class distribution:\n", y_val.value_counts())
    
    return X_train, X_val, y_train, y_val

def process_data(file_path, random_state):
    # Load and balance the dataset
    balanced_df = balance_dataset(file_path, random_state)
    X = balanced_df.drop(columns=['POLIMERO'])
    y = balanced_df['POLIMERO']
    
    # Balance the classes in training and validation datasets
    return balance_classes(X, y, random_state)

def test_on_hsi_encoded(model, file_path):
    # Load test dataset
    test_df = pd.read_excel(file_path)
    X_test = test_df.drop(columns=['POLIMERO'])
    y_test = test_df['POLIMERO']

    # Predict and evaluate
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    test_results = {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'confusion_matrix': cm
    }

    return test_results

def main():
    # Fixed parameters
    file_path_train = 'pellet_ecc_senza_duplicati.xlsx'
    file_path_test = 'HSI_encoded.xlsx'

    random_state = 9
    model_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 32
    }

    # Process and balance data
    X_train, X_val, y_train, y_val = process_data(file_path_train, random_state)

    # Train the model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # Validate the model
    y_val_pred = model.predict(X_val)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    val_cm = confusion_matrix(y_val, y_val_pred)

    print("\nValidation Results:")
    print("Precision:", val_report['weighted avg']['precision'])
    print("Recall:", val_report['weighted avg']['recall'])
    print("F1 Score:", val_report['weighted avg']['f1-score'])
    print("Confusion Matrix:\n", val_cm)

    # Test on HSI_encoded.xlsx
    test_results = test_on_hsi_encoded(model, file_path_test)
    print("\nTest Results on HSI_encoded.xlsx:")
    print("Precision:", test_results['precision'])
    print("Recall:", test_results['recall'])
    print("F1 Score:", test_results['f1_score'])
    print("Confusion Matrix:\n", test_results['confusion_matrix'])

if __name__ == '__main__':
    main()
