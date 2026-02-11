import numpy as np
import pandas as pd

ROWS = 10_000
COLS = 10_000
DTYPE = np.float32

DATASET_A = "acie_observational_10k_x_10k.csv"
DATASET_B = "acie_counterfactual_10k_x_10k.csv"

def structural_model(latent):
    """
    Simple causal propagation model:
    Observables derived from latent variables.
    """
    # Example nonlinear physical mapping
    obs = np.tanh(latent @ np.random.randn(latent.shape[1], 6000).astype(DTYPE))
    return obs.astype(DTYPE)

def generate_dataset(filename, intervention=False):
    print(f"Generating {filename}")

    chunk_size = 500
    first_chunk = True

    for start in range(0, ROWS, chunk_size):
        end = min(start + chunk_size, ROWS)
        print(f"Rows {start} to {end}")

        # --- LATENT PHYSICAL STATE (2000 dims) ---
        latent = np.random.normal(0, 1, size=(end - start, 2000)).astype(DTYPE)

        # Apply intervention for Dataset B
        if intervention:
            latent[:, 0:200] += 1.5  # do(Mass + delta)

        # --- OBSERVABLE PROPAGATION (6000 dims) ---
        observables = structural_model(latent)

        # --- NOISE / SELECTION BIAS (2000 dims) ---
        noise = np.random.normal(0, 0.1, size=(end - start, 2000)).astype(DTYPE)

        # Combine blocks → 10K features
        data = np.concatenate([latent, observables, noise], axis=1)
        
        # Convert to DataFrame and append to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, mode='a' if not first_chunk else 'w', 
                  header=first_chunk, index=False)
        first_chunk = False

    print("Done.\n")


if __name__ == "__main__":
    # Dataset A: Observational
    generate_dataset(DATASET_A, intervention=False)

    # Dataset B: Counterfactual
    generate_dataset(DATASET_B, intervention=True)
