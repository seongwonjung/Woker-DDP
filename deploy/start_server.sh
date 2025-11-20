#!/bin/bash
set -euo pipefail

APP_DIR="/home/ubuntu/app"
COMPOSE_FILE="$APP_DIR/docker-compose-prod.yml"
LOG_FILE="$APP_DIR/app.log"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "ERROR: docker compose file not found: $COMPOSE_FILE"
    exit 1
fi

cd "$APP_DIR"

echo "Stopping any existing docker compose log tail (if running)..."
pkill -f "docker compose -f $COMPOSE_FILE logs -f" >/dev/null 2>&1 || true

echo "Starting Docker stack with $COMPOSE_FILE..."
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans --build

echo "Streaming docker compose logs to $LOG_FILE..."
nohup docker compose -f "$COMPOSE_FILE" logs -f > "$LOG_FILE" 2>&1 &

echo "Life Cycle - ApplicationStart: complete."
