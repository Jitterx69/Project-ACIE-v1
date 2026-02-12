# Simplified Colab Training Guide

## 1. Prepare Minimal Zip
Run this locally to create `ACIE_Project.zip` containing ONLY the code needed for training:
```bash
./scripts/prepare_for_colab.sh
```

## 2. Drive Setup
1.  Go to Google Drive.
2.  Create a folder `ACIE`.
3.  Upload `ACIE_Project.zip` to `ACIE/`.
4.  Create a `data/` folder inside `ACIE/`.
5.  Upload your CSV files to `ACIE/data/`.

Structure should be:
```
My Drive/
└── ACIE/
    ├── ACIE_Project.zip
    └── data/
        └── acie_observational_10k_x_10k.csv
```

## 3. Run Notebook
1.  Upload `notebooks/ACIE_Colab_Training.ipynb` to Colab.
2.  Run the notebook.
3.  It will unzip the code to `/content/ACIE_Train` and link your data folder automatically.
