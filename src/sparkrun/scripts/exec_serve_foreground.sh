#!/bin/bash
set -uo pipefail

printf "Executing serve command in container '%s'...\n" "{container_name}"
echo "--- Command ---"
printf '%s' '{b64_cmd}' | base64 -d --
echo -e "\n---------------"

docker exec {container_name} bash -c "printf '%s' '{b64_cmd}' | base64 -d -- > /tmp/sparkrun_serve.sh && bash --noprofile --norc /tmp/sparkrun_serve.sh"
