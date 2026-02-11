import requests
import time
import subprocess
import sys
import os

def verify_api():
    print("🚀 Starting ACIE API Server in background...")
    # Start server in background
    proc = subprocess.Popen(
        [sys.executable, "acie/api/fastapi_server.py"],
        env=os.environ.copy()
    )
    
    try:
        # Wait for startup
        print("⏳ Waiting for server to start...")
        for _ in range(30):
            try:
                resp = requests.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    print("✅ Server is up!")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("❌ Server failed to start")
            return False

        # Check dashboard stats
        print("\n📊 Checking /api/dashboard/stats...")
        resp = requests.get("http://localhost:8080/api/dashboard/stats")
        
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Stats endpoint reachable")
            print(f"   Keys: {list(data.keys())}")
            print(f"   CPU Util: {data.get('cpu_util')}%")
            print(f"   GPU Count: {len(data.get('gpu', []))}")
            print(f"   Latency History: {len(data.get('latency_history', []))} items")
            return True
        else:
            print(f"❌ Failed to get stats: {resp.status_code} - {resp.text}")
            return False
            
    finally:
        print("\n🛑 Stopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    success = verify_api()
    sys.exit(0 if success else 1)
