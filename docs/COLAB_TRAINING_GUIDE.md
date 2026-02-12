# Minimal ACIE Training Guide (Google Colab)

This guide explains how to export **only the essential files** and train the model on Google Colab.

## Step 1: Export Minimal Project
Run this command on your local machine to create a lightweight zip file:

```bash
./scripts/prepare_for_colab.sh
```

This creates `ACIE_Project.zip` containing **only**:
*   `acie/` (Python source code)
*   `config/` (Configuration)
*   `setup.py` & `requirements.txt` (Dependencies)

## Step 2: Upload to Google Drive
1.  Go to [Google Drive](https://drive.google.com).
2.  Create a folder named **`ACIE_Training`**.
3.  Upload **`ACIE_Project.zip`** to this folder.
4.  Upload your **dataset CSVs** to this same folder.

**Your Drive structure should look exactly like this:**
```text
My Drive/
└── ACIE_Training/
    ├── ACIE_Project.zip
    ├── acie_observational_10k_x_10k.csv
    └── ... (other CSVs)
```

## Step 3: Start Training
1.  Open [Google Colab](https://colab.research.google.com).
2.  Upload `notebooks/ACIE_Colab_Training.ipynb`.
3.  **Run All Cells**.

The notebook will:
*   Find the `ACIE_Training` folder.
*   Unzip the code automatically.
*   Link your CSV files.
*   Start training.
