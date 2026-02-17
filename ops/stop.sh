#!/bin/bash
echo "=== Stopping ACIE Services ==="
docker-compose -f docker-compose.production.yml down
echo "=== Services Stopped ==="
