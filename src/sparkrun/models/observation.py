"""Exact-file host observations; no Python or Hub client is needed on targets."""

from __future__ import annotations

from sparkrun.models.artifacts import CacheValidationReport, ModelArtifactManifest, ModelFileFailure
from sparkrun.utils.shell import quote


def path_assignment(name: str, path: str) -> str:
    """Expand only the documented remote home prefixes, never arbitrary code."""
    if "\x00" in path:
        raise ValueError("NUL in model cache path")
    return (
        f"{name}={quote(path)}\n"
        f'case "${name}" in\n'
        f'  "~"|\'$HOME\') {name}="$HOME" ;;\n'
        f'  "~/"*) {name}="$HOME/${{{name}:2}}" ;;\n'
        f"  '$HOME/'*) {name}=\"$HOME/${{{name}:6}}\" ;;\n"
        "esac\n"
    )


def observation_script(manifest: ModelArtifactManifest, snapshot: str, *, checksums: bool = False, fail_on_error: bool = False) -> str:
    """Return one bounded batch, with ordinal IDs rather than parsed filenames."""
    arrays = {
        "names": [f.path for f in manifest.files],
        "expected": [str(f.size) for f in manifest.files],
        "algorithms": [f.checksum_algorithm or "none" for f in manifest.files],
        "checksums": [f.checksum or "none" for f in manifest.files],
    }
    script = "set -uo pipefail\nexport LC_ALL=C\n" + path_assignment("root", snapshot)
    script += "".join(name + "=(" + " ".join(quote(value) for value in values) + ")\n" for name, values in arrays.items())
    script += r"""
statuses=() paths=() ordinals=()
for i in "${!names[@]}"; do
    p="$root/${names[$i]}"
    s=ok
    if [ ! -e "$p" ]; then s=missing
    elif [ ! -f "$p" ]; then s=not_regular
    elif ! { : < "$p"; } 2>/dev/null; then s=unreadable
    else paths+=("$p"); ordinals+=("$i")
    fi
    statuses+=("$s")
done
# Batch metadata system calls without parsing names in stat output. xargs
# bounds argv size; NUL framing handles spaces, newlines and leading dashes.
if [ "${#paths[@]}" -gt 0 ]; then
    if observed=$(printf '%s\0' "${paths[@]}" | xargs -0 -r stat -L --printf='%s\n' -- 2>/dev/null); then
        mapfile -t sizes <<< "$observed"
        if [ "${#sizes[@]}" != "${#ordinals[@]}" ]; then
            for i in "${ordinals[@]}"; do statuses[$i]=unavailable; done
        else
            for n in "${!ordinals[@]}"; do
                i=${ordinals[$n]}
                if ! [[ "${sizes[$n]}" =~ ^[0-9]+$ ]]; then statuses[$i]=unavailable
                elif [ "${sizes[$n]}" != "${expected[$i]}" ]; then statuses[$i]=wrong_size
                fi
            done
        fi
    else
        # Mutation during the batch is not proof of a missing payload; refuse
        # to publish/repair based on a shifted or partial observation stream.
        for i in "${ordinals[@]}"; do statuses[$i]=unavailable; done
    fi
fi
"""
    if checksums:
        script += r"""
for i in "${!names[@]}"; do
    [ "${statuses[$i]}" = ok ] || continue
    p="$root/${names[$i]}"
    case "${algorithms[$i]}" in
        sha256) digest=$(sha256sum < "$p") || { statuses[$i]=unavailable; continue; } ;;
        git-sha1) digest=$( (printf 'blob %s\0' "${expected[$i]}"; cat -- "$p") | sha1sum) || { statuses[$i]=unavailable; continue; } ;;
        *) statuses[$i]=checksum_unavailable; continue ;;
    esac
    [ "${digest%% *}" = "${checksums[$i]}" ] || statuses[$i]=checksum_mismatch
done
"""
    script += r"""
failed=0
for i in "${!names[@]}"; do
    printf '%s\t%s\n' "$i" "${statuses[$i]}"
    [ "${statuses[$i]}" = ok ] || failed=1
done
printf 'END\t%d\n' "${#names[@]}"
"""
    if fail_on_error:
        script += 'exit "$failed"\n'
    return script


def parse_observations(
    host: str, manifest: ModelArtifactManifest, stdout: str, *, returncode: int = 0, checksums: bool = False
) -> CacheValidationReport:
    level = "checksum" if checksums else "metadata"
    if returncode:
        return CacheValidationReport(host, manifest.identity, level, "unavailable", detail="host observation failed (rc=%d)" % returncode)
    lines = stdout.splitlines()
    allowed = {"ok", "missing", "not_regular", "unreadable", "unavailable", "wrong_size", "checksum_mismatch", "checksum_unavailable"}
    if len(lines) != len(manifest.files) + 1 or lines[-1] != "END\t%d" % len(manifest.files):
        return CacheValidationReport(host, manifest.identity, level, "unavailable", detail="incomplete or malformed host observation")
    failures = []
    unknown = False
    for i, entry in enumerate(manifest.files):
        pair = lines[i].split("\t")
        if len(pair) != 2 or pair[0] != str(i) or pair[1] not in allowed:
            return CacheValidationReport(host, manifest.identity, level, "unavailable", detail="invalid or duplicate observation ID")
        if pair[1] != "ok":
            failures.append(ModelFileFailure(entry.path, pair[1]))
            unknown |= pair[1] in {"unavailable", "checksum_unavailable"}
    status = "complete" if not failures else "unverified" if unknown else "incomplete"
    return CacheValidationReport(host, manifest.identity, level, status, len(manifest.files), tuple(failures))
