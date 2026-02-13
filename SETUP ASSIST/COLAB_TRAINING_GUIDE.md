# Minimal ACIE Training Guide (Google Colab)

This guide explains how to export **only the essential files** and train the model on Google Colab.

## Step 1: Export Minimal Project
Run the script from the **SETUP ASSIST** folder:

```bash
./"SETUP ASSIST"/prepare_for_colab.sh
```

This creates `ACIE_Project.zip` inside the `SETUP ASSIST/` folder.

## Step 2: Upload to Google Drive
1.  Go to [Google Drive](https://drive.google.com).
2.  Create a folder named **`ACIE_Training`**.
3.  Upload **`ACIE_Project.zip`** (from `SETUP ASSIST/`) to this folder.
4.  Upload your **dataset CSVs** to this same folder.

**Your Drive structure should look exactly like this:**
```text
My Drive/
└── ACIE_Training/
    ├── ACIE_Project.zip
    ├── acie_observational_20k_x_20k.csv
    └── ... (other CSVs)
```

## Step 3: Start Training
1.  Open [Google Colab](https://colab.research.google.com).
2.  Upload `SETUP ASSIST/ACIE_Colab_Training.ipynb`.
3.  **Run All Cells**.

The notebook will:
*   Find the `ACIE_Training` folder.
*   Unzip the code automatically.
*   Link your CSV files.
*   Start training.
