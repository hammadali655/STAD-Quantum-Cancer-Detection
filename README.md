Cancer Detection Using Quantum AI on Sequential Molecular Data
=================================================================

Hybrid quantum‑classical pipeline for stomach adenocarcinoma (STAD) classification
using EIIP‑encoded amino acid sequences, PCA, SMOTE, Fisher feature selection,
and a 5‑model ensemble including Variational Quantum Circuits.



Data and models are archived on **Zenodo**:
https://doi.org/10.5281/zenodo.20085825



Overview
---------

This repository contains the complete source code for the research paper
“Cancer Detection Using Quantum AI on Sequential Molecular Data:
A Hybrid Quantum‑Classical Pipeline for Stomach Adenocarcinoma Classification”.

The pipeline processes 1,104 STAD‑annotated biological sequence samples from the
IntOGen database (https://www.intogen.org), balances them with SMOTE,
extracts the 8 most discriminative PCA components via Fisher scores, and trains
five architectures:

  - Classical Random Forest and Multi‑Layer Perceptron (baselines)
  - Standalone 8‑qubit Variational Quantum Circuit (PennyLane + PyTorch)
  - Hybrid Quantum‑MLP (16 Fisher PCA + 8 quantum expectation values → 24‑dim hybrid features)
  - Hybrid Quantum‑RF (16 PCA + 8 quantum expectation values)

Best performance is achieved by the Hybrid Q‑MLP with 93.22 % test accuracy
(AUC‑ROC = 0.9725, F1 = 0.9289, overfitting gap = 4.29 pp).



Repository Structure
---------------------

STAD-Quantum-Cancer-Detection/
  step0_encoding.py                   EIIP encoding of amino acid residues
  step1_pca_alt.py                    MinMax scaling + PCA (100 components)
  step1_check_dataset_pca.py          Dataset info (class distribution, etc.)
  step2_smote_augmentation.py         SMOTE balancing (1:1 ratio)
  step3_preprocessing_split_scale.py  Stratified train/test split + StandardScaler
  step4_fisher_feature_selection.py   Fisher score calculation (top‑8 features)
  step4_train_rf_pca.py              Classical Random Forest & MLP training
  step5_train_quantum_fisher.py      8‑qubit Variational Quantum Circuit (VQC)
  step6_train_hybrid_mlp_pca.py      Hybrid Q‑MLP (24 features, deep neural net)
  step6_train_hybrid_rf_pca.py       Hybrid Q‑RF (Random Forest on hybrid features)
  step7_final_evaluation.py          MLP / Quantum / Hybrid evaluation plots
  step8_final_evaluation.py          RF / Quantum / Hybrid evaluation plots
  step9_model_comparison.py          Full multi‑model comparison & figures
  app.py                             Interactive Streamlit frontend
  requirements.txt
  data/
    Mutated_Orignal_biosequence.csv   Raw input (small – the only data file in the repo)

All generated data (npy, csv), trained model weights (pkl, pth), reports and figures
are archived on Zenodo (see below).



Large Assets (Data, Models, Reports)
--------------------------------------

The processed datasets, trained model checkpoints, evaluation logs, and figures
are too large for GitHub and are permanently stored on Zenodo:

  https://doi.org/10.5281/zenodo.20085825

Step 1 – Download the Assets
  Go to the Zenodo link above and download STAD_quantum_assets_v1.0.zip (about 1 GB).

Step 2 – Extract into the Repository Root
  Unpack the ZIP directly into the root of this repository so that the following
  folders appear alongside the Python scripts:

  data/       EIIP encoded, PCA reduced, SMOTE balanced, train/test splits
  models/     Trained classifiers, scalers, PCA objects, Fisher indices
  reports/    Detailed training logs
  figures/    Performance graphs (ROC, bar charts, heatmaps, radars)

Step 3 – Install Dependencies
  pip install -r requirements.txt

Step 4 – Run the Application
  streamlit run app.py

  This launches the interactive web interface where you can paste a protein sequence
  and obtain a classification from all five models.

Step 5 (optional) – Reproduce All Results from Scratch
  To retrain every model from the original CSV, run the steps in numeric order:

  python step0_encoding.py
  python step1_pca_alt.py
  python step2_smote_augmentation.py
  python step3_preprocessing_split_scale.py
  python step4_fisher_feature_selection.py
  python step4_train_rf_pca.py
  python step5_train_quantum_fisher.py
  python step6_train_hybrid_mlp_pca.py
  python step6_train_hybrid_rf_pca.py
  python step9_model_comparison.py

  All generated files will be saved in data/, models/, reports/, and figures/
  just like the Zenodo archive.



Model Performance (Test Set, n = 413)
--------------------------------------

| Model               | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|---------------------|----------|---------|-----------|--------|----------|
| Random Forest       | 91.28 %  | 0.9683  | 0.94      | 0.88   | 0.91     |
| MLP                 | 91.53 %  | 0.9716  | 0.95      | 0.88   | 0.91     |
| Quantum VQC         | 80.63 %  | 0.8792  | 0.83      | 0.78   | 0.80     |
| Hybrid Q‑RF         | 90.56 %  | 0.9717  | 0.92      | 0.89   | 0.90     |
| Hybrid Q‑MLP        | 93.22 %  | 0.9725  | 0.98      | 0.88   | 0.93     |

Overfitting gap (train acc minus test acc) for the Hybrid Q‑MLP: 4.29 percentage points.



Requirements
------------

  - Python 3.9 or newer
  - Packages listed in requirements.txt
  - A modern CPU is sufficient for simulation (PennyLane + PyTorch).
    No actual quantum hardware is needed; the default.qubit simulator is used.



Citation
--------

If you use this code or the accompanying data, please cite:

  @article{ali2025quantum,
    title   = {Cancer Detection Using Quantum AI on Sequential Molecular Data:
               A Hybrid Quantum-Classical Pipeline for Stomach Adenocarcinoma Classification},
    author  = {Ali, Hammad and Ahmed, Hafeez and Sikander, Misba},
    journal = {to appear},
    year    = {2025},
    note    = {Code and data at https://github.com/hammadalib655/STAD-Quantum-Cancer-Detection
               and https://doi.org/10.5281/zenodo.20085825}
  }

(The journal details can be updated once the paper is published.)



Acknowledgements
-----------------

The authors extend their sincere gratitude to all respected faculty members of the IT
department, and especially to Dr. Misba Sikandar, whose extraordinary experience
and insightful supervision were instrumental in guiding this research to its successful
completion. The IntOGen consortium is also gratefully acknowledged for making somatic
mutation data publicly accessible.



Contact
-------

For questions or collaboration:
ha6262913@gmail.com
