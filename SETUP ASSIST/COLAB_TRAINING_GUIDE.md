# 🌌 ACIE Google Colab Training Guide

This comprehensive guide will help you set up and train the ACIE model on Google Colab using your specific datasets.

---

## 📂 Phase 1: Preparation (Local Machine)

### 1. Generate the Project Zip
We need to bundle the code into a lightweight package.

1.  Open your terminal.
2.  Navigate to the project root (`/Users/jitterx/Desktop/ACIE`).
3.  Run the preparation script:
    ```bash
    ./"SETUP ASSIST"/prepare_for_colab.sh
    ```
4.  **Result**: You will find a new file named `ACIE_Project.zip` inside the `SETUP ASSIST` folder.

---

## ☁️ Phase 2: Google Drive Setup

### 1. Create the Training Folder
1.  Go to [Google Drive](https://drive.google.com).
2.  Create a **new folder** in your "My Drive" named:
    👉 **`ACIE_Training`**
    *(The name must be exact for the notebook to work automatically)*

### 2. Upload Files
Upload the following files into the `ACIE_Training` folder:

| File Type | Filename | Source |
| :--- | :--- | :--- |
| **Project Code** | `ACIE_Project.zip` | From `SETUP ASSIST/` |
| **Dataset** | `acie_observational_20k_x_20k.csv` | From `lib/` |
| **Dataset** | `acie_hard_intervention_20k_x_20k.csv` | From `lib/` |
| **Dataset** | `acie_environment_shift_20k_x_20k.csv` | From `lib/` |
| **Dataset** | `acie_instrument_shift_20k_x_20k.csv` | From `lib/` |

> **Note**: We use the **20k** datasets because they include the intervention data required for robust training. The 10k datasets are incompatible with this training run due to dimension differences.

---

## 🚀 Phase 3: Train on Colab

### 1. Launch the Notebook
1.  Open [Google Colab](https://colab.research.google.com).
2.  Click **Upload**.
3.  Select the file: `SETUP ASSIST/ACIE_Colab_Training.ipynb`.

### 2. Configure Runtime
Before running anything:
1.  Go to the top menu: **Runtime** > **Change runtime type**.
2.  Hardware accelerator: **T4 GPU** (or better).
3.  Click **Save**.

### 3. Execution
1.  **Run Cell 1**: Mounts Google Drive. You will be asked to authorize access.
2.  **Run Cell 2**: Sets up the workspace and extracts your zip file.
3.  **Run Cell 3**: Links your CSV datasets. content should start with "✅ Linked: ...".
4.  **Run Cell 4**: Installs dependencies.
5.  **Run Cell 5**: **STARTS TRAINING**. You will see a progress bar for 20 epochs.
6.  **Run Cell 6**: Saves the trained model and logs back to `ACIE_Training/outputs/final_run`.

---

## ❓ Troubleshooting

**Q: "ACIE_Project.zip not found"**
*   Check that you uploaded it specifically to the `ACIE_Training` folder, not just "My Drive".
*   Check the filename is exactly `ACIE_Project.zip`.

**Q: "No valid 20k datasets found"**
*   Ensure you uploaded the CSVs ending in `_20k_x_20k.csv`.
*   The notebook is configured to look for these specifically to combine them into one large training set.

**Q: "Out of Memory (OOM)"**
*   If the session crashes, try reducing `BATCH_SIZE` in the training cell from `64` to `32`.
