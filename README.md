# polymer_characterization
A repository for pellets polymer characterization using Random Forest
In order to use it:
Install Python 3 on your system, check you have all libraries required (pandas, numpy, matplotlib and seaborn for confusion matrix production, scikit-learn for Random Forest, bayesian-optimization for model selection).
Galician dataset is contained in Galicia.xlsx. If you want to produce the encoding from scratch, it is necessary to discard manually from this Excel the first column and change names of each column in such a way to be compatible with code, accordingly (Italian or English). We suggest to use directly the encoded and cleaned files.
Vulcano dataset is contained in Vulcano.xlsx
If you want to use one of them to make an experiment, just run replace.py on the chosen one (replace its name inside the file). This file will make the ordinal encoding of the features.
In order to avoid duplicates inner-class and inter-classes, run duplication_removal.py on the chosen dataset (replace its name inside the file). Remind that you should run it after the replace.py, so if we call its output replaced_data.py, you should input replaced_data.py to duplication_removal.py (change the name inside the file).
For Galicia, we have already uploaded the encoded versions: Pellet_Stato_Ecc.xlsx is the encoded version of Galicia.xlsx, pellet_ecc_senza_duplicati.xlsx is the encoded version after duplication removal.
For Vulcano, we have already uploaded the encoded versions: HSI_encoded.xlsx.
Now, if you want to train and test on Galicia dataset, just run GalicianTrainingTest.py.
If you want, instead, to train on Galicia dataset and test on Vulcano dataset, just run HSI_test.py.
If you want to check the optimal parmeters for the RF, just run Bayesian.py, it will give the same target for many combinations, so what we did was choosing the average {100, 5, 2, 1} for the final model on 10% validation and then use it on test.
If you want ablation studies, just run ablations.py. The results are already contained in the file ablations.xlsx.

