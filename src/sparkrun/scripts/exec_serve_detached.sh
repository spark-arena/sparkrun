#!/bin/bash
set -uo pipefail

printf "Executing serve command in container '%s' (detached)...\n" "{container_name}"
echo "--- Command ---"
printf '%s' '{b64_cmd}' | base64 -d --
echo -e "\n---------------"

docker exec {container_name} bash -c "printf '%s' '{b64_cmd}' | base64 -d -- > /tmp/sparkrun_serve.sh && nohup bash --noprofile --norc /tmp/sparkrun_serve.sh > /tmp/sparkrun_serve.log 2>&1 & echo \$! > /tmp/sparkrun_serve.pid"

# Watchdog: when serve process exits, kill sleep infinity (PID 1) so container exits
docker exec -d {container_name} bash -c 'SERVE_PID=$(cat /tmp/sparkrun_serve.pid); while kill -0 $SERVE_PID 2>/dev/null; do sleep 5; done; kill 1'

# Wait for process to start and produce initial output
sleep 3

# Check if the serve process is still running
SERVE_PID=$(docker exec {container_name} cat /tmp/sparkrun_serve.pid 2>/dev/null)
if [ -n "$SERVE_PID" ] && ! docker exec {container_name} kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "============================================================" >&2
    echo "ERROR: Serve process exited immediately (PID $SERVE_PID)" >&2
    echo "Container logs:" >&2
    docker exec {container_name} cat /tmp/sparkrun_serve.log 2>/dev/null >&2 || true
    echo "============================================================" >&2
    exit 1
fi

echo "============================================================"
echo "Initial log output:"
docker exec {container_name} tail -n 30 /tmp/sparkrun_serve.log 2>/dev/null || echo "(no log output yet)"
echo "============================================================"
echo "Serve command launched in background."
echo "To follow logs:  ssh <host> docker exec {container_name} tail -f /tmp/sparkrun_serve.log"
