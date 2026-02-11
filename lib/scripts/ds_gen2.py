import numpy as np
import pandas as pd

ROWS = 20_000
COLS = 20_000
DTYPE = np.float32
CHUNK_SIZE = 500

DATASETS = {
    "observational": {"mass_shift": 0.0, "env_shift": 0.0, "noise_scale": 0.1},
    "hard_intervention": {"mass_shift": 2.0, "env_shift": 0.0, "noise_scale": 0.1},
    "environment_shift": {"mass_shift": 0.0, "env_shift": 1.5, "noise_scale": 0.1},
    "instrument_shift": {"mass_shift": 0.0, "env_shift": 0.0, "noise_scale": 0.5},
}

def structural_model(latent):
    W = np.random.randn(latent.shape[1], 11000).astype(DTYPE)
    obs = np.tanh(latent @ W)
    return obs.astype(DTYPE)

def generate_dataset(name, params):
    filename = f"acie_{name}_20k_x_20k.csv"
    print(f"Generating {filename}")

    first_chunk = True

    for start in range(0, ROWS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ROWS)
        print(f"{name}: rows {start} to {end}")

        # --- LATENT BLOCK (4000 dims) ---
        latent = np.random.normal(0, 1, size=(end - start, 4000)).astype(DTYPE)

        # Hard mass intervention
        latent[:, 0:400] += params["mass_shift"]

        # Environmental shift
        latent[:, 400:800] += params["env_shift"]

        # --- OBSERVABLE BLOCK (11000 dims) ---
        observables = structural_model(latent)

        # --- NOISE BLOCK (5000 dims) ---
        noise = np.random.normal(
            0,
            params["noise_scale"],
            size=(end - start, 5000)
        ).astype(DTYPE)

        data = np.concatenate([latent, observables, noise], axis=1)

        # Convert to DataFrame and append to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, mode='a' if not first_chunk else 'w',
                  header=first_chunk, index=False)
        first_chunk = False

    print(f"{name} completed.\n")


if __name__ == "__main__":
    for name, params in DATASETS.items():
        generate_dataset(name, params)
