from locust import HttpUser, task, between, constant
import random
import json

class ACIEUser(HttpUser):
    wait_time = constant(1)

    def on_start(self):
        """Called when a User starts"""
        pass

    @task(3)
    def inference(self):
        """Simulate inference request"""
        # Generate random observation vector
        observation = [random.random() for _ in range(10)]
        
        payload = {
            "observation": observation,
            "intervention": {"x": 1.0},
            "model_version": "latest"
        }
        
        with self.client.post("/api/v2/inference", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Simulate health check"""
        self.client.get("/health")
        
    @task(1)
    def metrics(self):
        """Check metrics endpoint"""
        self.client.get("/metrics")
