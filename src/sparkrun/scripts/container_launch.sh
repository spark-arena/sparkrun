#!/bin/bash
set -uo pipefail

printf "Cleaning up existing container: %%s\n" {container_name}
{cleanup_cmd}

printf "Launching container: %%s\n" {container_name}
printf "Image: %%s\n" {image}
{run_cmd}

printf "Container %%s launched successfully\n" {container_name}
