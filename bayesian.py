import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

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
    # Exclude 10% as test set first
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    # From the remaining, take 10% as validation
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1, random_state=random_state)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def optimize_rf(file_path):
    X, y = process_data(file_path)
    X_train, X_valid, _, y_train, y_valid, _ = balance_classes(X, y, random_state=9)
    
    fixed_params = {
        'max_features': 'sqrt'
    }

    def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        # Ensure min_samples_split is at least 2, as per the RandomForestClassifier constraints
        min_samples_split = max(int(min_samples_split), 2)  # Correct this parameter
        
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=min_samples_split,
            min_samples_leaf=int(min_samples_leaf),
            max_features='sqrt',
            random_state=32
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return accuracy_score(y_valid, y_pred)
    
    # Narrowing the parameter search range around reasonable values
    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds={
            'n_estimators': (95, 105),  # Narrow range around 100
            'max_depth': (4, 6),         # Narrow range around 5
            'min_samples_split': (2, 3), # Ensure this is >= 2
            'min_samples_leaf': (1, 2)   # Narrow range around 1
        },
        random_state=42
    )
    
    # Initialize the optimization
    optimizer.maximize(init_points=5, n_iter=50)
    
    optimal_params = optimizer.max['params']
    
    # Round results to integer values as needed for RandomForestClassifier
    optimal_params_rounded = {
        'n_estimators': int(round(optimal_params['n_estimators'])),
        'max_depth': int(round(optimal_params['max_depth'])),
        'min_samples_split': int(round(optimal_params['min_samples_split'])),
        'min_samples_leaf': int(round(optimal_params['min_samples_leaf']))
    }
    
    print(f'Optimal parameters found: {optimal_params_rounded}')
    print(f'Total iterations: {len(optimizer.res)}')

# Run optimization
file_path = 'pellet_ecc_senza_duplicati.xlsx'
optimize_rf(file_path)
