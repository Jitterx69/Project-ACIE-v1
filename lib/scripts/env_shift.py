import numpy as np
import pandas as pd

ROWS = 20_000
COLS = 20_000
CHUNK_SIZE = 200
DTYPE = np.float32

OUTPUT_FILE = "acie_environment_shift_20k_x_20k.csv"

LATENT_DIM = 4000
OBS_DIM = 11000
NOISE_DIM = 5000

def structural_model(latent, W):
    return np.tanh(latent @ W).astype(DTYPE)

def generate_environment_shift():
    print("Generating Environmental Shift Dataset...")

    # Fixed structural weights (important for consistency)
    W = np.random.randn(LATENT_DIM, OBS_DIM).astype(DTYPE)

    first_chunk = True

    for start in range(0, ROWS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ROWS)
        print(f"Rows {start} to {end}")

        # --- Latent Block ---
        latent = np.random.normal(0, 1, size=(end - start, LATENT_DIM)).astype(DTYPE)

        # Environmental shift applied
        latent[:, 400:800] += 1.5

        # --- Observables ---
        observables = structural_model(latent, W)

        # --- Noise ---
        noise = np.random.normal(0, 0.1, size=(end - start, NOISE_DIM)).astype(DTYPE)

        data = np.concatenate([latent, observables, noise], axis=1)

        # Convert to DataFrame and append to CSV
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, mode='a' if not first_chunk else 'w',
                  header=first_chunk, index=False)
        first_chunk = False

    print("Environmental Shift Dataset Complete.")

if __name__ == "__main__":
    generate_environment_shift()
