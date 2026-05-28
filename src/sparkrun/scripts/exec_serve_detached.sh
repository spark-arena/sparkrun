#!/bin/bash
set -uo pipefail

printf "Executing serve command in container '%s' (detached)...\n" "{container_name}"
echo "--- Command ---"
printf '%s' '{b64_cmd}' | base64 -d --
echo -e "\n---------------"

# Launch the serve command detached with stdout/stderr redirected to a file
# inside the container. Container PID 1 is `sleep infinity`, so `docker logs`
# is structurally blind to this output -- it must be read via `docker exec cat`.
docker exec {container_name} bash -c "printf '%s' '{b64_cmd}' | base64 -d -- > /tmp/sparkrun_serve.sh && nohup bash --noprofile --norc /tmp/sparkrun_serve.sh > /tmp/sparkrun_serve.log 2>&1 & echo \$! > /tmp/sparkrun_serve.pid"

# Wait for process to start and (hopefully) produce initial output
sleep 3

# Check if the serve process is still running
SERVE_PID=$(docker exec {container_name} cat /tmp/sparkrun_serve.pid 2>/dev/null)
if [ -n "$SERVE_PID" ] && ! docker exec {container_name} kill -0 "$SERVE_PID" 2>/dev/null; then
    # Failure path. The watchdog (below) hasn't been installed yet, so the
    # container is still alive and we can read the log. Python may have
    # buffered output that hasn't reached the file yet; poll briefly.
    LOG_CONTENT=""
    for _ in 1 2 3 4; do
        LOG_CONTENT=$(docker exec {container_name} cat /tmp/sparkrun_serve.log 2>/dev/null || true)
        if [ -n "$LOG_CONTENT" ]; then
            break
        fi
        sleep 0.5
    done

    echo "============================================================" >&2
    echo "ERROR: Serve process exited immediately (PID $SERVE_PID)" >&2
    echo "Container: {container_name}" >&2
    echo "Log file:  /tmp/sparkrun_serve.log (inside container)" >&2
    echo "------------------------------------------------------------" >&2
    if [ -n "$LOG_CONTENT" ]; then
        printf '%s\n' "$LOG_CONTENT" >&2
    elif ! docker exec {container_name} test -e /tmp/sparkrun_serve.log 2>/dev/null; then
        echo "(log file does not exist -- container may have exited or serve never started)" >&2
    else
        echo "(log file empty after ~5s; process likely crashed before producing output)" >&2
        echo "To inspect manually:  docker exec {container_name} cat /tmp/sparkrun_serve.log" >&2
    fi
    echo "============================================================" >&2
    exit 1
fi

# Success path. Install the watchdog so the container exits whenever the
# serve process dies. We install it AFTER the liveness check so that, on
# early failure, the container stays alive long enough to read the log file.
docker exec -d {container_name} bash -c 'SERVE_PID=$(cat /tmp/sparkrun_serve.pid); while kill -0 $SERVE_PID 2>/dev/null; do sleep 5; done; kill 1'

echo "============================================================"
echo "Initial log output:"
docker exec {container_name} tail -n 30 /tmp/sparkrun_serve.log 2>/dev/null || echo "(no log output yet)"
echo "============================================================"
echo "Serve command launched in background."
echo "To follow logs:  ssh <host> docker exec {container_name} tail -f /tmp/sparkrun_serve.log"
