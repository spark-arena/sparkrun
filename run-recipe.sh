#!/usr/bin/env bash
#
# run-recipe.sh - spark-vllm-docker compatibility shim, backed by `sparkrun`.
#
# This accepts the same CLI surface as the legacy spark-vllm-docker
# `run-recipe.py`/`run-recipe.sh` pair, but performs the work through the
# modern `sparkrun` CLI instead. It exists so existing spark-vllm-docker
# invocations keep working while users migrate to sparkrun.
#
# Runner resolution (first hit wins):
#   1. <repo>/.venv/bin/sparkrun       (local editable/dev install)
#   2. `sparkrun` on PATH              (active venv / system install)
#   3. `uv tool run sparkrun`          (uvx; uv is installed via system pip
#                                       if it is not already available)
#
# INTENTIONAL DEVIATIONS FROM THE LEGACY TOOL (see also the plan file):
#   * `--list` lists sparkrun *registry* recipes, not a local recipes/ dir.
#   * Engine passthrough after `--` is translated to `-o key=value` (sparkrun's
#     only sanctioned engine-override path), not appended verbatim. It only
#     takes effect for recipe-templated / known engine keys.
#   * No autodiscovery/.env workflow: --discover / --show-env / --config error out.
#   * --build-only / --download-only have no isolated phase under sparkrun
#     (images/models are synced automatically during `run`).
#
# Hidden testing hook:
#   RUN_RECIPE_DEBUG=1  -> print the assembled `sparkrun` argv to stderr and
#                          exit 0 WITHOUT invoking sparkrun (lets the mapping be
#                          asserted even when sparkrun/uv are not installed).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG="$(basename "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
err()  { printf '%s\n' "$*" >&2; }
die()  { err "error: $*"; exit "${2:-1}"; }

# Hard error for a legacy option that has no faithful sparkrun equivalent.
unsupported() {
    # $1 = option label, $2 = native pointer
    err "error: ${1} is not supported by the ${PROG} sparkrun shim."
    err "       ${2}"
    exit 2
}

print_banner() {
    err "┌─ ${PROG} (sparkrun compatibility shim) ─────────────────────────────┐"
    err "│ This wraps \`sparkrun\` for spark-vllm-docker-style invocations.            │"
    err "│ For full control over placement, schedulers, registries, tuning, proxy,   │"
    err "│ and many more options, call \`sparkrun run --help\` directly.               │"
    err "└───────────────────────────────────────────────────────────────────────────┘"
}

usage() {
    cat >&2 <<EOF
${PROG} - run a model recipe via sparkrun (spark-vllm-docker-compatible CLI)

Usage:
  ${PROG} RECIPE [options] [-- <engine passthrough args>]
  ${PROG} --list

Common options (mapped onto \`sparkrun run\`):
  --port N                      Override serve port
  --host ADDR                   Override bind address (-> -o host=ADDR)
  --tp, --tensor-parallel N     Tensor parallelism
  --gpu-mem, --gpu-memory-utilization F
  --max-model-len N
  -n, --nodes a,b,c             Cluster hosts (-> --hosts)
  -t, --container IMG           Override image (-> --image)
  --name NAME                   Container name (-> --container-name)
  --solo                        Single-node mode (implies --tp 1 if unset)
  -d, --daemon                  Detach (default here is foreground)
  --no-ray                      -> -o distributed_executor_backend=mp
  -e, --env VAR=VAL             Container env var (repeatable)
  -p, --publish H:C             Publish port, solo only (repeatable)
  --nccl-debug LEVEL            -> -e NCCL_DEBUG=LEVEL
  --dry-run                     Show the plan without launching

Run \`sparkrun run --help\` for the full, native option set.
EOF
}

# ---------------------------------------------------------------------------
# Resolve how we will invoke sparkrun  ->  SPARKRUN=(...)
# ---------------------------------------------------------------------------
resolve_runner() {
    # 1. Local repo dev install.
    if [[ -x "${SCRIPT_DIR}/.venv/bin/sparkrun" ]]; then
        SPARKRUN=("${SCRIPT_DIR}/.venv/bin/sparkrun")
        return
    fi
    # 2. sparkrun already on PATH.
    if command -v sparkrun >/dev/null 2>&1; then
        SPARKRUN=(sparkrun)
        return
    fi
    # 3. Fall back to uv (installing uv via system pip if necessary).
    if ! command -v uv >/dev/null 2>&1; then
        local py=""
        if command -v python3 >/dev/null 2>&1; then
            py=python3
        elif command -v python >/dev/null 2>&1; then
            py=python
        else
            die "no local sparkrun, and no python3/python to bootstrap uv. Install uv (https://docs.astral.sh/uv/) or sparkrun."
        fi
        err "${PROG}: uv not found; installing uv via ${py} -m pip ..."
        if ! "${py}" -m pip install --user uv >&2; then
            # --user can fail inside an active venv; retry without it.
            "${py}" -m pip install uv >&2 \
                || die "failed to install uv via pip. Install uv manually: https://docs.astral.sh/uv/"
        fi
        export PATH="${HOME}/.local/bin:${PATH}"
        if command -v uv >/dev/null 2>&1; then
            SPARKRUN=(uv tool run sparkrun)
            return
        fi
        # uv binary not on PATH even after install -> use `python -m uv`.
        if "${py}" -m uv --version >/dev/null 2>&1; then
            SPARKRUN=("${py}" -m uv tool run sparkrun)
            return
        fi
        die "uv was installed but could not be located on PATH. Add ~/.local/bin to PATH and retry."
    fi
    SPARKRUN=(uv tool run sparkrun)
}

# ---------------------------------------------------------------------------
# Parse the legacy CLI surface and build the sparkrun `run` argv (ARGS).
# ---------------------------------------------------------------------------
RECIPE=""
WANT_LIST=0
SOLO=0
NO_RAY=0
DAEMON=0
TP_SET=0
HAVE_PUBLISH=0
declare -a ARGS=()              # sparkrun run arguments (after the recipe)
declare -a PASSTHROUGH=()       # tokens after `--`

# consume_value: resolve the value for an option that takes one.
#   Reads globals: HAS_EQ, EQ_VAL, NEXT_TOK, HAVE_NEXT.
#   Sets globals:  VAL  (the value), DBL (1 if a second positional must be shifted).
# $1 = option label (for error messages)
consume_value() {
    if [[ $HAS_EQ -eq 1 ]]; then
        VAL="$EQ_VAL"; DBL=0; return 0
    fi
    [[ $HAVE_NEXT -eq 1 ]] || die "$1 requires a value"
    VAL="$NEXT_TOK"; DBL=1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        local arg="$1"
        HAS_EQ=0
        EQ_VAL=""
        NEXT_TOK="${2:-}"
        HAVE_NEXT=0
        [[ $# -ge 2 ]] && HAVE_NEXT=1
        VAL=""
        DBL=0

        # Split --opt=value forms (only long options use =).
        case "$arg" in
            --*=*)
                EQ_VAL="${arg#*=}"
                arg="${arg%%=*}"
                HAS_EQ=1
                ;;
        esac

        case "$arg" in
            # ----- list / help -----
            -l|--list)   WANT_LIST=1; shift; continue ;;
            -h|--help)   usage; exit 0 ;;

            # ----- end-of-options: rest is engine passthrough -----
            --) shift; PASSTHROUGH=("$@"); break ;;

            # ----- directly mapped value options -----
            --port)
                consume_value "--port"; ARGS+=(--port "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --host)
                consume_value "--host"; ARGS+=(-o "host=${VAL}"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --tp|--tensor-parallel)
                consume_value "--tp"; ARGS+=(--tp "$VAL"); TP_SET=1; shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --gpu-mem|--gpu-memory-utilization)
                consume_value "--gpu-mem"; ARGS+=(--gpu-mem "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --max-model-len)
                consume_value "--max-model-len"; ARGS+=(--max-model-len "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            -n|--nodes)
                consume_value "--nodes"; ARGS+=(--hosts "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            -t|--container)
                consume_value "--container"; ARGS+=(--image "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --name)
                consume_value "--name"; ARGS+=(--container-name "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --master-port|--head-port)
                consume_value "--master-port"; ARGS+=(--init-port "$VAL"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            -e|--env)
                consume_value "--env"; ARGS+=(--executor-args "-e ${VAL}"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --nccl-debug)
                consume_value "--nccl-debug"; ARGS+=(--executor-args "-e NCCL_DEBUG=${VAL}"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            -p|--publish)
                consume_value "--publish"; ARGS+=(--executor-args "-p ${VAL}"); HAVE_PUBLISH=1; shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --mem-limit-gb)
                consume_value "--mem-limit-gb"; ARGS+=(--memory-limit "${VAL}G"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --shm-size-gb)
                consume_value "--shm-size-gb"; ARGS+=(-o "shm_size=${VAL}g"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --mem-swap-limit-gb)
                consume_value "--mem-swap-limit-gb"; ARGS+=(--executor-args "--memory-swap ${VAL}g"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;
            --pids-limit)
                consume_value "--pids-limit"; ARGS+=(--executor-args "--pids-limit ${VAL}"); shift; [[ $DBL -eq 1 ]] && shift; continue ;;

            # ----- boolean / mode flags -----
            --solo)            SOLO=1; ARGS+=(--solo); shift; continue ;;
            --no-ray)          NO_RAY=1; ARGS+=(-o distributed_executor_backend=mp); shift; continue ;;
            -d|--daemon)       DAEMON=1; shift; continue ;;
            --dry-run)         ARGS+=(--dry-run); shift; continue ;;
            --non-privileged)  ARGS+=(-o privileged=false); shift; continue ;;
            --setup)
                err "${PROG}: --setup is a no-op; sparkrun syncs images/models automatically during run."
                shift; continue ;;

            # ----- hard-errored legacy options -----
            --apply-mod)       unsupported "--apply-mod" "mods are a v1/eugr concept; bake changes into the recipe." ;;
            --eth-if)          unsupported "--eth-if"    "sparkrun auto-detects networking; no manual interface override." ;;
            --ib-if)           unsupported "--ib-if"     "sparkrun auto-detects networking; no manual interface override." ;;
            --discover)        unsupported "--discover"  "use \`sparkrun setup wizard\` or \`sparkrun cluster create\`." ;;
            --show-env)        unsupported "--show-env"  "use \`sparkrun cluster show <name>\`." ;;
            --config)          unsupported "--config"    "use \`--cluster NAME\`, \`-n/--nodes\`, or a hosts file." ;;
            --build-only)      unsupported "--build-only"    "sparkrun syncs images/models during run; no isolated phase." ;;
            --download-only)   unsupported "--download-only" "sparkrun syncs images/models during run; no isolated phase." ;;
            --force-build)     unsupported "--force-build"    "no build/download phase to force; re-pull the image manually if needed." ;;
            --force-download)  unsupported "--force-download" "no build/download phase to force; re-download the model manually if needed." ;;
            -j)                unsupported "-j" "no in-container build step under sparkrun." ;;
            --keep-entrypoint) unsupported "--keep-entrypoint" "sparkrun manages the container entrypoint." ;;
            --no-cache-dirs)   unsupported "--no-cache-dirs" "sparkrun manages cache mounts." ;;

            # ----- unknown options -----
            -*) die "unknown option: ${arg} (run \`sparkrun run --help\` for native options)" ;;

            # ----- positional recipe -----
            *)
                if [[ -n "$RECIPE" ]]; then
                    die "unexpected extra argument: ${arg} (recipe already set to '${RECIPE}')"
                fi
                RECIPE="$arg"; shift; continue ;;
        esac
    done
}

# Translate post-`--` engine passthrough tokens into `-o key=value`.
translate_passthrough() {
    [[ ${#PASSTHROUGH[@]} -eq 0 ]] && return 0
    err "${PROG}: note: engine passthrough after \`--\` is mapped to \`-o key=value\`;"
    err "       it only takes effect for recipe-templated / known engine keys (unlike the legacy verbatim append)."
    local i=0 n=${#PASSTHROUGH[@]}
    while [[ $i -lt $n ]]; do
        local tok="${PASSTHROUGH[$i]}"
        case "$tok" in
            --*=*)
                local k="${tok%%=*}"; local v="${tok#*=}"
                k="${k#--}"; k="${k//-/_}"
                ARGS+=(-o "${k}=${v}")
                ;;
            --*)
                local k="${tok#--}"; k="${k//-/_}"
                local nxt=""
                if [[ $((i + 1)) -lt $n ]]; then nxt="${PASSTHROUGH[$((i + 1))]}"; fi
                if [[ $((i + 1)) -lt $n && "$nxt" != -* ]]; then
                    ARGS+=(-o "${k}=${nxt}")
                    i=$((i + 1))
                else
                    ARGS+=(-o "${k}=true")
                fi
                ;;
            *)
                err "${PROG}: warning: ignoring bare passthrough token '${tok}' (no leading --)." ;;
        esac
        i=$((i + 1))
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    print_banner

    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    parse_args "$@"

    # --list short-circuits everything else (matches legacy precedence).
    if [[ $WANT_LIST -eq 1 ]]; then
        resolve_runner
        if [[ "${RUN_RECIPE_DEBUG:-0}" == "1" ]]; then
            err "DEBUG argv: ${SPARKRUN[*]} recipe list"
            exit 0
        fi
        exec "${SPARKRUN[@]}" recipe list
    fi

    if [[ -z "$RECIPE" ]]; then
        die "no recipe specified. Run \`${PROG} --help\` for usage."
    fi

    # ----- pre-flight compatibility checks (mirror the legacy tool) -----
    if [[ $NO_RAY -eq 1 && $SOLO -eq 1 ]]; then
        die "--no-ray is incompatible with --solo (solo already runs without Ray)."
    fi
    if [[ $HAVE_PUBLISH -eq 1 && $SOLO -eq 0 ]]; then
        die "-p/--publish is only supported in solo mode; add --solo or drop the port mapping."
    fi

    # Solo defaults tensor_parallel=1 unless the user set --tp.
    if [[ $SOLO -eq 1 && $TP_SET -eq 0 ]]; then
        ARGS+=(--tp 1)
    fi

    # Foreground is the legacy default; -d/--daemon opts into sparkrun's detach.
    if [[ $DAEMON -eq 0 ]]; then
        ARGS+=(--foreground)
    fi

    translate_passthrough

    resolve_runner

    if [[ "${RUN_RECIPE_DEBUG:-0}" == "1" ]]; then
        err "DEBUG argv: ${SPARKRUN[*]} run ${RECIPE} ${ARGS[*]}"
        exit 0
    fi

    exec "${SPARKRUN[@]}" run "$RECIPE" "${ARGS[@]}"
}

main "$@"
