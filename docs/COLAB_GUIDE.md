# ACIE Google Colab Training Guide

This guide explains how to train the ACIE model using Google Colab's free GPU resources.

## Step 1: Prepare Your Code
We need to package your code to upload it to Google Drive.

1.  Open a terminal in your project root.
2.  Run the preparation script:
    ```bash
    chmod +x scripts/prepare_for_colab.sh
    ./scripts/prepare_for_colab.sh
    ```
3.  This will create `ACIE_Project.zip`.

## Step 2: Upload to Google Drive
1.  Go to [Google Drive](https://drive.google.com).
2.  Create a folder named `ACIE` (or any name you prefer).
3.  Upload `ACIE_Project.zip` to this folder.
4.  **Important**: Upload your CSV datasets (e.g., `acie_observational_10k_x_10k.csv`) to a `data/` subfolder inside `ACIE`.

## Step 3: Open the Notebook
1.  Go to [Google Colab](https://colab.research.google.com).
2.  Click **Upload** -> **Browse** and select `notebooks/ACIE_Colab_Training.ipynb` from your local project.
3.  The notebook will open.

## Step 4: Configure & Run
1.  **Set Runtime**: Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU`.
2.  **Mount Drive**: Run the first few cells to mount your Google Drive.
3.  **Unzip**: The notebook has a cell to unzip your project. Uncomment and run it if you haven't unzipped it manually in Drive.
    ```python
    !unzip "/content/drive/My Drive/ACIE/ACIE_Project.zip" -d /content/ACIE_Train
    ```
4.  **Install & Train**: Run the remaining cells to install dependencies and start training.

## Tips
-   **Persistent Storage**: The notebook is configured to save outputs to `outputs/`. You may want to copy these back to Drive periodically to avoid losing them if the runtime disconnects.
-   **GPU**: Always ensure you are connected to a GPU instance (`!nvidia-smi` should show a GPU).
