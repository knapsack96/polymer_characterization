import pandas as pd

# Carica il dataset
file_path = 'Pellet_Stato_Ecc.xlsx'  # Sostituisci con il percorso corretto
df = pd.read_excel(file_path)

# Separare i dati in base a POLIMERO
pp_records = df[df['POLIMERO'] == 'PP'].drop(columns=['POLIMERO'])
pe_records = df[df['POLIMERO'] == 'PE'].drop(columns=['POLIMERO'])

# 1. Rimuovi duplicati all'interno della stessa classe (PP e PE)
pp_records_unique = pp_records.drop_duplicates()
pe_records_unique = pe_records.drop_duplicates()

# 2. Rimuovi duplicati tra le classi (PP e PE) che sono uguali tranne che per POLIMERO
duplicates_between_classes = []

# Verifica se esistono duplicati tra i record di PP e PE
for idx_pp, row_pp in pp_records_unique.iterrows():
    for idx_pe, row_pe in pe_records_unique.iterrows():
        if row_pp.equals(row_pe):  # Se i record sono identici tranne che per POLIMERO
            duplicates_between_classes.append((idx_pp, idx_pe))

# Rimuovi i duplicati tra le classi, mantenendo il record nella classe più piccola
for pp_idx, pe_idx in duplicates_between_classes:
    # Se la classe PP è più piccola, rimuovi il duplicato da PE
    if len(pp_records_unique) < len(pe_records_unique):
        pe_records_unique = pe_records_unique.drop(pe_idx)
    else:
        # Altrimenti, rimuovi il duplicato da PP
        pp_records_unique = pp_records_unique.drop(pp_idx)

# Ripristina la colonna 'POLIMERO' nelle variabili uniche
pp_records_unique['POLIMERO'] = 'PP'
pe_records_unique['POLIMERO'] = 'PE'

# 3. Contare duplicati all'interno della stessa classe (PP e PE)
pp_duplicates = pp_records_unique.duplicated(subset=pp_records_unique.columns).sum()
pe_duplicates = pe_records_unique.duplicated(subset=pe_records_unique.columns).sum()

# 4. Contare duplicati tra le due classi
duplicates_between_classes_count = len(duplicates_between_classes)

# Stampa i risultati
print(f"Duplicati all'interno della classe PP: {pp_duplicates}")
print(f"Duplicati all'interno della classe PE: {pe_duplicates}")
print(f"Duplicati tra le classi (PP e PE): {duplicates_between_classes_count}")

# Per vedere la distribuzione finale di PP e PE dopo la rimozione dei duplicati
df_unique = pd.concat([pp_records_unique, pe_records_unique])
distribution = df_unique['POLIMERO'].value_counts()

print("\nDistribuzione dei polimeri dopo la rimozione dei duplicati:")
print(distribution)

# 5. Ricalcola di nuovo i duplicati dopo la rimozione
pp_duplicates_final = pp_records_unique.duplicated(subset=pp_records_unique.columns).sum()
pe_duplicates_final = pe_records_unique.duplicated(subset=pe_records_unique.columns).sum()

print(f"\nDuplicati finali all'interno della classe PP: {pp_duplicates_final}")
print(f"Duplicati finali all'interno della classe PE: {pe_duplicates_final}")

# Salva il dataframe risultante (senza duplicati) in un file Excel
output_file = 'pellet_ecc_senza_duplicati.xlsx'
df_unique.to_excel(output_file, index=False)

print(f"\nIl dataset senza duplicati è stato salvato come '{output_file}'.")
