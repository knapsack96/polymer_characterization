import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations
import seaborn as sns

def balance_dataset(file_path, arg=9):
    df = pd.read_excel(file_path)
    filtered_df = df[df['POLIMERO'].isin(['PP', 'PE'])]
    pp_samples = filtered_df[filtered_df['POLIMERO'] == 'PP']
    pe_samples = filtered_df[filtered_df['POLIMERO'] == 'PE']
    
    pp_balanced = pp_samples.sample(n=531, random_state=arg) if len(pp_samples) >= 531 else pp_samples
    pe_balanced = pe_samples.sample(n=531, random_state=arg) if len(pe_samples) >= 531 else pe_samples
    
    balanced_df = pd.concat([pp_balanced, pe_balanced])
    return balanced_df

def process_data(file_path, arg=9):
    balanced_df = balance_dataset(file_path, arg)
    X = balanced_df.drop(columns=['POLIMERO'])
    y = balanced_df['POLIMERO']
    return X, y

def balance_classes(X, y, random_state):
    X_class_0 = X[y == 'PE']
    y_class_0 = y[y == 'PE']
    
    X_class_1 = X[y == 'PP']
    y_class_1 = y[y == 'PP']
    
    X_class_0_train, X_class_0_test, y_class_0_train, y_class_0_test = train_test_split(X_class_0, y_class_0, test_size=0.1, random_state=random_state)
    X_class_1_train, X_class_1_test, y_class_1_train, y_class_1_test = train_test_split(X_class_1, y_class_1, test_size=0.1, random_state=random_state)
    
    X_train = pd.concat([X_class_0_train, X_class_1_train])
    y_train = pd.concat([y_class_0_train, y_class_1_train])
    
    X_test = pd.concat([X_class_0_test, X_class_1_test])
    y_test = pd.concat([y_class_0_test, y_class_1_test])
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrices_in_grid(matrices, class_names, filenames):
    n_matrices = len(matrices)
    n_rows = (n_matrices // 3) + (n_matrices % 3 != 0)  # Number of rows, 3 per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    axes = axes.ravel()

    for i, (cm, filename) in enumerate(zip(matrices, filenames)):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[],
                    cbar=True, ax=ax, annot_kws={"size": 20})  # Adjusted font size for values

        ax.set_xlabel('Predicted PE    Predicted PP', fontsize=20)  # Increased font size for x-label
        ax.set_ylabel('True PP                 True PE', fontsize=20)  # Increased font size for y-label
        
        # Title only with the subset variables
        title = filename.replace("_", ", ").replace(".jpg", "").replace("confusion, matrix, ","")  # Remove extra characters
        ax.set_title(title, fontsize=20)
        
        # Add a heatmap colorbar showing min and max values
        cbar = ax.collections[0].colorbar
        cbar.set_label('Count', fontsize=18)  # Increased font size for colorbar label
        cbar.ax.tick_params(labelsize=18)  # Adjusted font size for colorbar ticks
    
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('confusion_matrices_grid.jpg', dpi=300, format='jpg')  # Save as a grid of confusion matrices
    plt.close()

def evaluate_models(file_path):
    results = []
    
    fixed_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'max_features': 'sqrt',
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    balance_random_state = 9
    X, y = process_data(file_path, balance_random_state)
    X_train, X_test, y_train, y_test = balance_classes(X, y, balance_random_state)
    
    model = RandomForestClassifier(random_state=32, **fixed_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    
    # Classi da utilizzare per le etichette
    class_names = model.classes_
    print(class_names)
    matrices = []
    filenames = []
    
    # Per ogni combinazione di sottoinsiemi di feature, calcoliamo e salviamo la matrice di confusione
    feature_names = X.columns
    for size in range(1, len(feature_names) + 1):
        for subset in combinations(feature_names, size):
            X_subset = X_test[list(subset)].copy()
            
            # Calcolare la previsione per il sottoinsieme di variabili
            model.fit(X_train[list(subset)], y_train)
            y_pred_subset = model.predict(X_test[list(subset)])
            
            # Calcolare la matrice di confusione
            cm_subset = confusion_matrix(y_test, y_pred_subset)
            
            # Create a filename for the matrix
            subset_name = "_".join(subset)
            filename = f"confusion_matrix_{subset_name}.jpg"
            
            matrices.append(cm_subset)
            filenames.append(filename)
    
    # Plot and save all confusion matrices in a grid
    plot_confusion_matrices_in_grid(matrices, class_names, filenames)

    return results

# Main call
file_path = 'pellet_ecc_senza_duplicati.xlsx'
evaluate_models(file_path)

print("Confusion matrices grid saved as 'confusion_matrices_grid.jpg'")
