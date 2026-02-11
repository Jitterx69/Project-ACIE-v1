"""
Enhanced CLI for ACIE using Typer

Modern command-line interface with:
- Rich console output with colors and formatting
- Progress bars and spinners
- Better error messages
- Interactive prompts
- Auto-completion support
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer(
    name="acie",
    help="🌌 ACIE - Astronomical Counterfactual Inference Engine",
    add_completion=True,
)

console = Console()


@app.command()
def train(
    data_dir: Path = typer.Option(..., help="Directory containing training data"),
    output_dir: Path = typer.Option(..., help="Directory for outputs"),
    obs_dim: int = typer.Option(6000, help="Observable dimension"),
    latent_dim: int = typer.Option(2000, help="Latent dimension"),
    batch_size: int = typer.Option(128, help="Batch size"),
    max_epochs: int = typer.Option(100, help="Maximum training epochs"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    dataset_size: str = typer.Option("10k", help="Dataset size (10k or 20k)"),
    no_counterfactual: bool = typer.Option(False, help="Don't use counterfactual data"),
    gpus: int = typer.Option(1, help="Number of GPUs to use"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    fast_dev_run: bool = typer.Option(False, help="Quick test run"),
):
    """
    🚀 Train ACIE model with PyTorch Lightning
    
    Example:
        acie train --data-dir ./data --output-dir ./outputs
    """
    from acie.training.train import train_acie
    
    console.print(Panel.fit(
        f"[bold cyan]Training ACIE Model[/bold cyan]\n\n"
        f"Dataset: {dataset_size}\n"
        f"Data directory: {data_dir}\n"
        f"Output directory: {output_dir}\n"
        f"GPUs: {gpus}",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing training...", total=None)
        
        try:
            model, trainer = train_acie(
                data_dir=data_dir,
                output_dir=output_dir,
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                dataset_size=dataset_size,
                use_counterfactual=not no_counterfactual,
                gpus=gpus,
                num_workers=num_workers,
                fast_dev_run=fast_dev_run,
            )
            progress.update(task, description="[green]✓ Training complete!")
            console.print("[bold green]✓ Training successful![/bold green]")
            
        except Exception as e:
            progress.update(task, description="[red]✗ Training failed")
            console.print(f"[bold red]✗ Error: {e}[/bold red]")
            raise typer.Exit(code=1)


@app.command()
def infer(
    checkpoint: Path = typer.Option(..., help="Model checkpoint path"),
    observation_file: Path = typer.Option(..., help="File with observations"),
    intervention: str = typer.Option(..., help="Intervention (e.g., 'mass=1.5,metallicity=0.02')"),
    output_dir: Path = typer.Option("results", help="Output directory"),
):
    """
    🔮 Perform counterfactual inference
    
    Example:
        acie infer --checkpoint model.ckpt --observation-file obs.csv --intervention "mass=1.5"
    """
    from acie.training.train import ACIELightningModule
    import pandas as pd
    import torch
    
    console.print(f"[cyan]Loading model from[/cyan] {checkpoint}")
    
    try:
        model = ACIELightningModule.load_from_checkpoint(str(checkpoint))
        model.eval()
        
        # Load observation
        if str(observation_file).endswith('.csv'):
            obs_data = pd.read_csv(observation_file)
            obs = torch.tensor(obs_data.values, dtype=torch.float32)
        else:
            obs = torch.load(observation_file)
        
        console.print(f"[cyan]Observation shape:[/cyan] {obs.shape}")
        
        # Parse intervention
        intervention_dict = {}
        for item in intervention.split(','):
            var, val = item.split('=')
            intervention_dict[var.strip()] = float(val.strip())
        
        console.print(f"[cyan]Intervention:[/cyan] {intervention_dict}")
        
        # Perform inference
        with Progress(SpinnerColumn(), TextColumn("[cyan]Computing counterfactual..."), console=console):
            engine = model.get_acie_engine()
            counterfactual = engine.intervene(obs, intervention_dict)
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(counterfactual, output_dir / "counterfactual.pt")
        pd.DataFrame(counterfactual.cpu().numpy()).to_csv(
            output_dir / "counterfactual.csv", index=False
        )
        
        console.print(f"[bold green]✓ Results saved to {output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., help="Model checkpoint path"),
    data_dir: Path = typer.Option(..., help="Test data directory"),
    dataset_size: str = typer.Option("10k", help="Dataset size (10k or 20k)"),
    batch_size: int = typer.Option(128, help="Batch size"),
    max_samples: Optional[int] = typer.Option(None, help="Maximum samples to evaluate"),
    device: str = typer.Option("cuda", help="Device to use (cuda/mps/cpu)"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for results"),
):
    """
    📊 Evaluate trained model on test data
    
    Example:
        acie evaluate --checkpoint model.ckpt --data-dir ./data
    """
    from acie.training.train import ACIELightningModule
    from acie.data.dataset import create_dataloaders
    from acie.eval.metrics import CounterfactualEvaluator
    import json
    
    console.print(f"[cyan]Loading model from[/cyan] {checkpoint}")
    
    try:
        model = ACIELightningModule.load_from_checkpoint(str(checkpoint))
        model.eval()
        
        # Load test data
        with Progress(SpinnerColumn(), TextColumn("[cyan]Loading test data..."), console=console):
            dataloaders = create_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                dataset_size=dataset_size,
                max_rows=max_samples,
            )
        
        test_loader = dataloaders.get("counterfactual") or dataloaders.get("observational")
        
        if test_loader is None:
            console.print("[bold red]✗ No test data found![/bold red]")
            raise typer.Exit(code=1)
        
        # Evaluate
        console.print("[cyan]Evaluating model...[/cyan]")
        evaluator = CounterfactualEvaluator()
        results = evaluator.evaluate(model, test_loader, device=device)
        
        # Print results
        evaluator.print_results()
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[bold green]✓ Results saved to {output_dir / 'evaluation_results.json'}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    checkpoint: Path = typer.Option(..., help="Model checkpoint path"),
    num_samples: int = typer.Option(100, help="Number of samples to benchmark"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("cuda", help="Device to use (cuda/mps/cpu)"),
):
    """
    ⚡ Benchmark model inference performance
    
    Example:
        acie benchmark --checkpoint model.ckpt --num-samples 1000
    """
    from acie.training.train import ACIELightningModule
    import torch
    import time
    import numpy as np
    
    console.print(Panel.fit(
        f"[bold cyan]ACIE Performance Benchmark[/bold cyan]\n\n"
        f"Checkpoint: {checkpoint}\n"
        f"Samples: {num_samples}\n"
        f"Batch size: {batch_size}\n"
        f"Device: {device}",
        border_style="cyan"
    ))
    
    try:
        # Load model
        with console.status("[cyan]Loading model..."):
            model = ACIELightningModule.load_from_checkpoint(str(checkpoint))
            model.eval()
            model = model.to(device)
        
        # Generate random data
        obs_dim = 6000  # Default ACIE obs dim
        test_data = torch.randn(num_samples, obs_dim).to(device)
        
        latencies = []
        
        # Warmup
        with console.status("[yellow]Warming up..."):
            _ = model(test_data[:batch_size])
        
        # Benchmark
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Running benchmark...", total=num_samples // batch_size)
            
            for i in range(0, num_samples, batch_size):
                batch = test_data[i:i+batch_size]
                
                start = time.time()
                with torch.no_grad():
                    _ = model(batch)
                if device in ['cuda', 'mps']:
                    torch.cuda.synchronize() if device == 'cuda' else None
                latency = (time.time() - start) * 1000  # ms
                
                latencies.append(latency / len(batch))  # Per-sample latency
                progress.update(task, advance=1)
        
        # Results
        table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total samples", str(num_samples))
        table.add_row("Mean latency", f"{np.mean(latencies):.2f} ms/sample")
        table.add_row("Median latency", f"{np.median(latencies):.2f} ms/sample")
        table.add_row("P95 latency", f"{np.percentile(latencies, 95):.2f} ms/sample")
        table.add_row("P99 latency", f"{np.percentile(latencies, 99):.2f} ms/sample")
        table.add_row("Throughput", f"{1000 / np.mean(latencies):.0f} samples/sec")
        
        console.print(table)
        console.print("[bold green]✓ Benchmark complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def serve(
    checkpoint: Path = typer.Option(..., help="Model checkpoint path"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to bind to"),
    workers: int = typer.Option(4, help="Number of worker processes"),
):
    """
    🌐 Start FastAPI inference server
    
    Example:
        acie serve --checkpoint model.ckpt --port 8080
    """
    console.print(Panel.fit(
        f"[bold cyan]Starting ACIE API Server[/bold cyan]\n\n"
        f"Checkpoint: {checkpoint}\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}",
        border_style="cyan"
    ))
    
    try:
        import uvicorn
        from acie.api.fastapi_server import app as fastapi_app
        
        console.print(f"[cyan]Server will be available at[/cyan] http://{host}:{port}")
        console.print(f"[cyan]API docs at[/cyan] http://{host}:{port}/docs")
        
        uvicorn.run(
            "acie.api.fastapi_server:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def export(
    checkpoint: Path = typer.Option(..., help="Model checkpoint path"),
    output_path: Path = typer.Option(..., help="Output path for exported model"),
    format: str = typer.Option("onnx", help="Export format (onnx/torchscript)"),
    opset_version: int = typer.Option(14, help="ONNX opset version"),
):
    """
    💾 Export model to ONNX or TorchScript
    
    Example:
        acie export --checkpoint model.ckpt --output-path model.onnx --format onnx
    """
    from acie.training.train import ACIELightningModule
    import torch
    
    console.print(f"[cyan]Loading model from[/cyan] {checkpoint}")
    
    try:
        model = ACIELightningModule.load_from_checkpoint(str(checkpoint))
        model.eval()
        
        # Create dummy input
        batch_size = 1
        obs_dim = 6000
        dummy_input = torch.randn(batch_size, obs_dim)
        
        output_path = Path(output_path)
        
        if format == "onnx":
            console.print(f"[cyan]Exporting to ONNX (opset {opset_version})...[/cyan]")
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                input_names=['observation'],
                output_names=['counterfactual'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'counterfactual': {0: 'batch_size'}
                }
            )
        elif format == "torchscript":
            console.print("[cyan]Exporting to TorchScript...[/cyan]")
            scripted = torch.jit.script(model)
            scripted.save(str(output_path))
        else:
            console.print(f"[bold red]✗ Unknown format: {format}[/bold red]")
            raise typer.Exit(code=1)
        
        console.print(f"[bold green]✓ Model exported to {output_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def version():
    """
    📦 Show ACIE version and system information
    """
    import torch
    import platform
    
    table = Table(title="ACIE System Information", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Version/Info", style="green")
    
    table.add_row("ACIE", "0.1.0")
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA available", str(torch.cuda.is_available()))
    table.add_row("MPS available", str(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    table.add_row("Platform", platform.platform())
    
    console.print(table)



# ============================================================================
# MLOps Commands (Phase 6)
# ============================================================================

@app.command()
def experiments(
    list_runs: bool = typer.Option(False, "--list", "-l", help="List recent experiments"),
    limit: int = typer.Option(10, help="Number of runs to list"),
):
    """
    🧪 Manage MLflow experiments
    """
    import mlflow
    from datetime import datetime
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    if list_runs:
        console.print(Panel.fit("[bold cyan]Recent Experiments[/bold cyan]", border_style="cyan"))
        
        try:
            experiments = mlflow.search_experiments()
            for exp in experiments:
                console.print(f"[bold]Experiment: {exp.name}[/bold] (ID: {exp.experiment_id})")
                
                # List runs for this experiment
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=limit,
                    order_by=["start_time DESC"]
                )
                
                if runs.empty:
                    console.print("  [dim]No runs found[/dim]")
                    continue
                    
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Run ID", style="cyan", no_wrap=True)
                table.add_column("Date", style="green")
                table.add_column("Status", style="yellow")
                
                for _, run in runs.iterrows():
                    start_time = datetime.fromtimestamp(run.start_time.timestamp()).strftime('%Y-%m-%d %H:%M')
                    table.add_row(run.run_id, start_time, run.status)
                    
                console.print(table)
                console.print()
                
        except Exception as e:
            console.print(f"[bold red]✗ Error: {e}[/bold red]")
            raise typer.Exit(code=1)
    else:
        console.print("Use --list to show experiments")


@app.command()
def register(
    run_id: str = typer.Option(..., help="MLflow Run ID to register"),
    name: str = typer.Option(..., help="Name for the registered model"),
    artifact_path: str = typer.Option("model", help="Path to model artifact"),
):
    """
    📚 Register a model from an experiment run
    """
    from acie.tracking.registry import ModelRegistry
    
    console.print(f"[cyan]Registering model from run[/cyan] {run_id}")
    
    try:
        registry = ModelRegistry()
        version = registry.register_model(run_id, name, artifact_path)
        
        console.print(Panel.fit(
            f"[bold green]✓ Model Registered Successfully[/bold green]\n\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            f"Run ID: {run_id}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def promote(
    name: str = typer.Option(..., help="Registered model name"),
    version: str = typer.Option(..., help="Model version to promote"),
    stage: str = typer.Option(..., help="Target stage (Staging/Production/Archived)"),
):
    """
    🚀 Promote a model version to a specific stage
    """
    from acie.tracking.registry import ModelRegistry
    
    console.print(f"[cyan]Promoting[/cyan] {name} v{version} [cyan]to[/cyan] {stage}")
    
    try:
        registry = ModelRegistry()
        registry.promote_model(name, version, stage)
        
        console.print(f"[bold green]✓ Model promoted to {stage}![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        raise typer.Exit(code=1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
