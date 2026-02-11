"""
Model export utilities (TorchScript, ONNX)
"""

import torch
import torch.onnx
from typing import Tuple, Union, Optional
import os
import logging

logger = logging.getLogger(__name__)


def export_to_torchscript(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    save_path: str,
    method: str = "trace"
) -> torch.jit.ScriptModule:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        example_input: Example input tensor(s) for tracing
        save_path: Path to save the exported model
        method: Export method ('trace' or 'script')
        
    Returns:
        Exported TorchScript module
    """
    model.eval()
    
    try:
        if method == "trace":
            logger.info("Tracing model...")
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(save_path)
            logger.info(f"Model exported to {save_path}")
            return traced_model
            
        elif method == "script":
            logger.info("Scripting model...")
            scripted_model = torch.jit.script(model)
            scripted_model.save(save_path)
            logger.info(f"Model exported to {save_path}")
            return scripted_model
            
        else:
            raise ValueError(f"Unknown export method: {method}")
            
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        raise


def export_to_onnx(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    save_path: str,
    input_names: list = ["input"],
    output_names: list = ["output"],
    opset_version: int = 12,
    dynamic_axes: Optional[dict] = None
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        example_input: Example input tensor
        save_path: Path to save the exported model
        input_names: List of input names
        output_names: List of output names
        opset_version: ONNX opset version
        dynamic_axes: Dictionary defining dynamic axes
    """
    model.eval()
    
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
        
    try:
        logger.info(f"Exporting to ONNX: {save_path}")
        torch.onnx.export(
            model,
            example_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        logger.info("ONNX export completed")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise
