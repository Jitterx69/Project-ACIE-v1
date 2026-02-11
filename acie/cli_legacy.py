"""
Command-Line Interface for ACIE

Commands:
- train: Train ACIE model
- infer: Perform counterfactual inference
- evaluate: Evaluate trained model
- query: Interactive counterfactual queries
"""

import argparse
from pathlib import Path
import torch


def train_command(args):
    """Train ACIE model."""
    from acie.training.train import train_acie
    
    print(f"Training ACIE on {args.dataset_size} dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    model, trainer = train_acie(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        obs_dim=args.obs_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        dataset_size=args.dataset_size,
        use_counterfactual=not args.no_counterfactual,
        gpus=args.gpus,
        num_workers=args.num_workers,
        fast_dev_run=args.fast_dev_run,
    )
    
    print("\nTraining complete!")


def infer_command(args):
    """Perform counterfactual inference."""
    from acie.training.train import ACIELightningModule
    from acie.data.dataset import ACIEDataset
    import pandas as pd
    
    print(f"Loading model from {args.checkpoint}")
    model = ACIELightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Load observation
    if args.observation_file.endswith('.csv'):
        obs_data = pd.read_csv(args.observation_file)
        obs = torch.tensor(obs_data.values, dtype=torch.float32)
    else:
        obs = torch.load(args.observation_file)
    
    print(f"Observation shape: {obs.shape}")
    
    # Parse intervention
    intervention = {}
    for item in args.intervention.split(','):
        var, val = item.split('=')
        intervention[var.strip()] = float(val.strip())
    
    print(f"Intervention: {intervention}")
    
    # Get ACIE engine
    engine = model.get_acie_engine()
    
    # Perform intervention
    counterfactual = engine.intervene(obs, intervention)
    
    print(f"Counterfactual shape: {counterfactual.shape}")
    
    # Save result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(counterfactual, output_dir / "counterfactual.pt")
    pd.DataFrame(counterfactual.cpu().numpy()).to_csv(
        output_dir / "counterfactual.csv", index=False
    )
    
    print(f"Results saved to {output_dir}")


def evaluate_command(args):
    """Evaluate trained model."""
    from acie.training.train import ACIELightningModule
    from acie.data.dataset import create_dataloaders
    from acie.eval.metrics import CounterfactualEvaluator
    
    print(f"Loading model from {args.checkpoint}")
    model = ACIELightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Load test data
    dataloaders = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        max_rows=args.max_samples,
    )
    
    test_loader = dataloaders.get("counterfactual") or dataloaders.get("observational")
    
    if test_loader is None:
        print("Error: No test data found!")
        return
    
    # Evaluate
    evaluator = CounterfactualEvaluator()
    results = evaluator.evaluate(model, test_loader, device=args.device)
    
    # Print results
    evaluator.print_results()
    
    # Save results
    if args.output_dir:
        import json
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_dir / 'evaluation_results.json'}")


def query_command(args):
    """Interactive counterfactual query."""
    from acie.training.train import ACIELightningModule
    
    print("Loading ACIE engine...")
    model = ACIELightningModule.load_from_checkpoint(args.checkpoint)
    engine = model.get_acie_engine()
    
    print("\nACIE Interactive Query Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get observation
        obs_input = input("Enter observation file path (or 'quit'): ").strip()
        if obs_input.lower() == 'quit':
            break
        
        try:
            import pandas as pd
            obs_data = pd.read_csv(obs_input)
            obs = torch.tensor(obs_data.values, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading observation: {e}")
            continue
        
        # Get intervention
        intervention_input = input("Enter intervention (e.g., 'mass=1.5,metallicity=0.02'): ").strip()
        
        intervention = {}
        for item in intervention_input.split(','):
            try:
                var, val = item.split('=')
                intervention[var.strip()] = float(val.strip())
            except:
                print(f"Invalid intervention format: {item}")
                continue
        
        # Perform query
        print("\nComputing counterfactual...")
        result = engine.counterfactual_query(obs, intervention)
        
        print(f"\nResults:")
        print(f"  Factual observation shape: {result['factual_obs'].shape}")
        print(f"  Counterfactual observation shape: {result['counterfactual_obs'].shape}")
        print(f"  Mean change: {(result['counterfactual_obs'] - result['factual_obs']).abs().mean().item():.6f}")
        
        # Save option
        save = input("\nSave results? (y/n): ").strip().lower()
        if save == 'y':
            output_path = input("Output path: ").strip()
            torch.save(result, output_path)
            print(f"Saved to {output_path}")
        
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACIE - Astronomical Counterfactual Inference Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ACIE model")
    train_parser.add_argument("--data-dir", type=str, required=True,
                             help="Directory containing training data")
    train_parser.add_argument("--output-dir", type=str, required=True,
                             help="Directory for outputs")
    train_parser.add_argument("--obs-dim", type=int, default=6000,
                             help="Observable dimension")
    train_parser.add_argument("--latent-dim", type=int, default=2000,
                             help="Latent dimension")
    train_parser.add_argument("--batch-size", type=int, default=128,
                             help="Batch size")
    train_parser.add_argument("--max-epochs", type=int, default=100,
                             help="Maximum epochs")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4,
                             help="Learning rate")
    train_parser.add_argument("--dataset-size", type=str, default="10k",
                             choices=["10k", "20k"], help="Dataset size")
    train_parser.add_argument("--no-counterfactual", action="store_true",
                             help="Don't use counterfactual data")
    train_parser.add_argument("--gpus", type=int, default=1,
                             help="Number of GPUs")
    train_parser.add_argument("--num-workers", type=int, default=4,
                             help="DataLoader workers")
    train_parser.add_argument("--fast-dev-run", action="store_true",
                             help="Quick test run")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Perform inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Model checkpoint path")
    infer_parser.add_argument("--observation-file", type=str, required=True,
                             help="File with observations")
    infer_parser.add_argument("--intervention", type=str, required=True,
                             help="Intervention (e.g., 'mass=1.5')")
    infer_parser.add_argument("--output-dir", type=str, default="results",
                             help="Output directory")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Model checkpoint")
    eval_parser.add_argument("--data-dir", type=str, required=True,
                            help="Test data directory")
    eval_parser.add_argument("--dataset-size", type=str, default="10k",
                            choices=["10k", "20k"])
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--max-samples", type=int, default=None,
                            help="Maximum samples to evaluate")
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--output-dir", type=str, default=None)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Interactive queries")
    query_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Model checkpoint")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "query":
        query_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
