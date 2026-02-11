"""
ACIE Client - Python client for ACIE Inference API

Simple interface for interacting with ACIE services.
"""

import requests
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin
import time

from acie_sdk.exceptions import (
    ACIEConnectionError,
    ACIEAuthenticationError,
    ACIENotFoundError,
    ACIEServerError,
    ACIETimeoutError,
    ACIEValidationError,
)


class ACIEClient:
    """
    Client for ACIE Inference API.
    
    Example:
        >>> client = ACIEClient("http://localhost:8080", api_key="your-key")
        >>> result = client.infer([1.0, 2.0, 3.0], {"mass": 1.5})
        >>> print(result['counterfactual'])
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """
        Initialize ACIE client.
        
        Args:
            api_url: Base URL of ACIE API (e.g., "http://localhost:8080")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Whether to verify SSL certificates
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "acie-sdk/1.0.0",
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retries and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/api/v2/inference")
            data: Request body data
            params: URL parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            ACIEConnectionError: Connection failed
            ACIEAuthenticationError: Authentication failed
            ACIENotFoundError: Resource not found
            ACIEServerError: Server error
            ACIETimeoutError: Request timed out
        """
        url = urljoin(self.api_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                
                # Handle different status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise ACIEAuthenticationError("Authentication failed. Check your API key.")
                elif response.status_code == 404:
                    raise ACIENotFoundError(f"Resource not found: {endpoint}")
                elif response.status_code == 422:
                    raise ACIEValidationError(f"Validation error: {response.text}")
                elif response.status_code >= 500:
                    raise ACIEServerError(
                        f"Server error: {response.text}",
                        status_code=response.status_code,
                        response_data=response.json() if response.text else None
                    )
                else:
                    raise ACIEServerError(
                        f"Unexpected status {response.status_code}: {response.text}",
                        status_code=response.status_code
                    )
                    
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise ACIETimeoutError(f"Request timed out after {self.timeout}s")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise ACIEConnectionError(f"Failed to connect to {self.api_url}: {e}")
                time.sleep(2 ** attempt)
    
    def infer(
        self,
        observation: List[float],
        intervention: Dict[str, float],
        model_version: str = "latest",
    ) -> Dict[str, Any]:
        """
        Perform counterfactual inference.
        
        Args:
            observation: List of observed values
            intervention: Dictionary of interventions (e.g., {"mass": 1.5})
            model_version: Model version to use
            
        Returns:
            Dictionary containing:
                - counterfactual: Counterfactual observation
                - latent_state: Latent representation
                - confidence: Confidence score
                
        Example:
            >>> result = client.infer([1.0, 2.0], {"mass": 1.5})
            >>> print(result['counterfactual'])
        """
        data = {
            "observation": observation,
            "intervention": intervention,
            "model_version": model_version,
        }
        
        return self._request("POST", "/api/v2/inference/counterfactual", data=data)
    
    def batch_infer(
        self,
        observations: List[List[float]],
        interventions: List[Dict[str, float]],
        model_version: str = "latest",
    ) -> List[Dict[str, Any]]:
        """
        Perform batch counterfactual inference.
        
        Args:
            observations: List of observations
            interventions: List of interventions
            model_version: Model version to use
            
        Returns:
            List of inference results
        """
        data = {
            "observations": observations,
            "interventions": interventions,
            "model_version": model_version,
        }
        
        return self._request("POST", "/api/v2/inference/batch", data=data)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.
        
        Returns:
            Health status information
        """
        return self._request("GET", "/health")
    
    def get_model_info(self, model_version: str = "latest") -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_version: Model version
            
        Returns:
            Model metadata and statistics
        """
        return self._request("GET", f"/api/v2/models/{model_version}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of available models with metadata
        """
        return self._request("GET", "/api/v2/models")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Returns:
            System performance metrics
        """
        return self._request("GET", "/api/metrics")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
    
    def close(self):
        """Close the client session."""
        self.session.close()
