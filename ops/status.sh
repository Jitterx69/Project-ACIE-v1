#!/bin/bash
echo "=== ACIE Service Status ==="
docker-compose -f docker-compose.production.yml ps
