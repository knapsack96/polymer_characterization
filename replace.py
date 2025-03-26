import pandas as pd

def process_and_replace(file_path):
    # Leggi il file Excel e seleziona il foglio "Polimero"
    df = pd.read_excel(file_path)

    df['POLIMERO'] = df['POLIMERO'].replace('pp', 'PP')

    
    # Rimuovi i record con NaN nella colonna "PESO"
    df = df.dropna(subset=['PESO'])

    
    
    # Sostituisci i valori in "STATO"
    stato_mapping = {
        'Non degradato': 1,
        
        'Molto degradato': 2
    }
    df['STATO'] = df['STATO'].replace(stato_mapping)

    # Sostituisci i valori in "COLORE"
    colore_mapping = {
        'Trasparente':1, 'Bianco':2, 'Arancione':3, 'Nero':4, 'bianco':2, 'Verde':5, 'Azzurro':6
    }
    df['COLORE'] = df['COLORE'].replace(colore_mapping)

    # Prima sostituisci "Piccolo" con "Piccola" in "DIMENSIONE"
    df['DIMENSIONE'] = df['DIMENSIONE'].replace('Piccolo', 'Piccola')

    # Poi sostituisci i valori in "DIMENSIONE"
    dimensione_mapping = {
        'Piccola': 1,
        'Medio': 2,
        'Grande': 3
    }
    df['DIMENSIONE'] = df['DIMENSIONE'].replace(dimensione_mapping)

    
    # Conta le occorrenze per ogni valore di "POLIMERO"
    polymer_counts = df['POLIMERO'].value_counts()

    # Salva il DataFrame modificato in un nuovo file Excel (opzionale)
    df.to_excel('HSI_encoded.xlsx', index=False)

    return polymer_counts

# Esempio di utilizzo:
file_path = 'HSI.xlsx'  # Sostituisci con il percorso del tuo file Excel
polymer_counts = process_and_replace(file_path)

# Stampa le occorrenze per ogni valore di "POLIMERO"
print("Occorrenze per ogni valore di 'POLIMERO':")
print(polymer_counts)
