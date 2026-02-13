# ACIE SDK

Python client library for ACIE (Astronomical Counterfactual Inference Engine).

## Installation

```bash
pip install acie[sdk]
```

Or for development:

```bash
pip install -e ".[sdk]"
```

## Quick Start

```python
from acie_sdk import ACIEClient

# Initialize client
client = ACIEClient("http://localhost:8080", api_key="your-key")

# Health check
health = client.health_check()
print(health)

# Perform counterfactual inference
observation = [1.0, 2.0, 3.0, ...]  # Your observation data
intervention = {"mass": 1.5, "metallicity": 0.02}

result = client.infer(observation, intervention)
print(result['counterfactual'])
print(result['confidence'])

# Close client
client.close()
```

## Features

- ✅ Simple HTTP client for ACIE API
- ✅ Automatic retries with exponential backoff
- ✅ Authentication support
- ✅ Batch inference
- ✅ Context manager support
- ✅ Comprehensive error handling

## Examples

See [examples/](../examples/) for detailed usage:
- `sdk_quickstart.py` - Basic SDK usage
- `sdk_advanced.py` - Advanced features (batch inference, monitoring)

## API Reference

### `ACIEClient`

Main client class for interacting with ACIE API.

#### Methods

- **`infer(observation, intervention, model_version="latest")`** - Single inference
- **`batch_infer(observations, interventions, model_version="latest")`** - Batch inference
- **`health_check()`** - Check server health
- **`get_model_info(model_version="latest")`** - Get model metadata
- **`list_models()`** - List available models
- **`get_metrics()`** - Get system metrics

#### Parameters

- `api_url` (str): Base URL of ACIE API
- `api_key` (str, optional): API key for authentication
- `timeout` (int): Request timeout in seconds (default: 30)
- `max_retries` (int): Maximum retry attempts (default: 3)
- `verify_ssl` (bool): Verify SSL certificates (default: True)

## Error Handling

The SDK provides specific exception types:

```python
from acie_sdk import ACIEClient, ACIEError, ACIEConnectionError

try:
    client = ACIEClient("http://localhost:8080")
result = client.infer(observation, intervention)
except ACIEConnectionError:
    print("Failed to connect to server")
except ACIEError as e:
    print(f"API error: {e}")
```

Exception hierarchy:
- `ACIEError` - Base exception
  - `ACIEConnectionError` - Connection failed
  - `ACIEAuthenticationError` - Authentication failed
  - `ACIENotFoundError` - Resource not found
  - `ACIEServerError` - Server error
  - `ACIETimeoutError` - Request timeout
  - `ACIEValidationError` - Validation error

## Context Manager

Use as context manager for automatic cleanup:

```python
with ACIEClient("http://localhost:8080") as client:
    result = client.infer(observation, intervention)
# Client automatically closed
```

## License

Same as ACIE project.
