import numpy as np
import pandas as pd

ROWS = 20_000
COLS = 20_000
CHUNK_SIZE = 200
DTYPE = np.float32

OUTPUT_FILE = "acie_instrument_shift_20k_x_20k.csv"

LATENT_DIM = 4000
OBS_DIM = 11000
NOISE_DIM = 5000

def structural_model(latent, W):
    return np.tanh(latent @ W).astype(DTYPE)

def generate_instrument_shift():
    print("Generating Instrument Shift Dataset...")

    W = np.random.randn(LATENT_DIM, OBS_DIM).astype(DTYPE)

    with open(OUTPUT_FILE, "w") as f:
        for start in range(0, ROWS, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, ROWS)
            print(f"Rows {start} to {end}")

            # --- Latent Block ---
            latent = np.random.normal(0, 1, size=(end - start, LATENT_DIM)).astype(DTYPE)

            # --- Observables ---
            observables = structural_model(latent, W)

            # --- Increased Instrument Noise ---
            noise = np.random.normal(0, 0.5, size=(end - start, NOISE_DIM)).astype(DTYPE)

            data = np.concatenate([latent, observables, noise], axis=1)

            np.savetxt(f, data, delimiter=",", fmt="%.6f")

    print("Instrument Shift Dataset Complete.")

if __name__ == "__main__":
    generate_instrument_shift()

