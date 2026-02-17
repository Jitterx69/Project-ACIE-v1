#!/bin/bash
echo "=== Starting ACIE Services ==="
docker-compose -f docker-compose.production.yml up -d --build
echo "=== Services Started ==="
echo "Frontend: http://localhost"
echo "API Docs: http://localhost/docs"
echo "Java API: http://localhost/api/java/"
echo "Grafana:  http://localhost:3000"
