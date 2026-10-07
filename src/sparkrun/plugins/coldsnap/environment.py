# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Capture-time recipe interpolation and artifact-owned restore environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import sparkrun.api as api
from sparkrun.core.env_templates import (
    LAUNCH_FIELDS,
    env_template_launch_fields,
    probe_model_path,
    render_env_template,
    workload_model_path,
)
from sparkrun.orchestration.primitives import build_ssh_kwargs
from sparkrun.plugins.coldsnap.artifacts import _read_committed_artifact
from sparkrun.plugins.coldsnap.runtime_cache import CANONICAL_RUNTIME_CACHE_ROOT

# This non-transport key is persisted and compared by existing controllers.
# Keep declared-input identity separate from concrete values already captured
# inside a process. A fresh container env cannot reconfigure an n610 process.
ENV_INPUTS = "SPARKRUN_COLDSNAP_ENV_INPUTS"


def validate_environment_recipe(recipe) -> None:
    if recipe.pre_exec or recipe.post_exec or recipe.post_commands or recipe.mods:
        raise ValueError(
            "ColdSnap does not support recipe hooks or mods; prepare the image/assets through "
            "ColdSnap before capture (ordinary pre_exec is not run during capture or restore)"
        )
    if ENV_INPUTS in recipe.env:
        raise ValueError(ENV_INPUTS + " is reserved for ColdSnap environment identity")


class RecipeEnvironment:
    """Supply materialize() with resolved template keys, never ordinary probes on restore."""

    def __init__(self, operation, options, *, plan, sctx, artifact="", deferred=False):
        validate_environment_recipe(plan.recipe)
        self.operation, self.options, self.plan, self.sctx = operation, options, plan, sctx
        self.templates = plan.recipe.env_templates
        self.config = plan.recipe.build_config_chain(options.overrides)
        self.fields = env_template_launch_fields(self.templates)
        self.deferred = deferred
        self.models = {}
        self.captured = {}
        if operation != "capture" and not deferred:
            path = Path(artifact).expanduser()
            if path.is_dir():
                path /= "artifact.json"
            # Literal-only legacy requests can still be rendered before their
            # artifact arrives. Templates require their committed provenance.
            if self.templates or path.is_file():
                try:
                    _, document = _read_committed_artifact(path.resolve())
                except RuntimeError:
                    if self.templates:
                        raise
                    # Leave legacy artifact diagnostics to the existing service
                    # and controller validation when no templates are requested.
                else:
                    units = document.get("launch", {}).get("units", [])
                    self.captured = {unit["id"]: unit.get("environment", {}) for unit in units}
        if self.fields & {"model_path", "model_revision"}:
            recipe = plan.recipe
            if self.config.get("model") != recipe.model or self.config.get("model_revision") != recipe.model_revision:
                raise ValueError("ColdSnap launch model fields must match the prepared recipe model and model_revision")
        if "runtime_cache_dir" in self.fields:
            policy = plan.recipe.plugin_item("coldsnap")
            root = PurePosixPath(CANONICAL_RUNTIME_CACHE_ROOT)
            if not policy.cache_seed or not any(root.is_relative_to(path) for path in policy.cache_paths):
                raise ValueError("launch.runtime_cache_dir requires ColdSnap cache seeding at " + str(root))

    def _placement(self, host):
        return {
            "num_nodes": len(self.plan.host_list),
            "node_rank": self.plan.host_list.index(host),
            "node_host": host,
            "cluster_id": self.plan.cluster_id,
        }

    def _identity(self, host):
        # Rendering only config references validates scalars and avoids hashing
        # unrelated defaults. Placement bindings are included only when used;
        # arbitrary host/cluster env values cannot be transparently rebound.
        symbolic = {name: "{launch." + name + "}" for name in LAUNCH_FIELDS}
        payload = {
            "templates": self.templates,
            "configured": {key: render_env_template(value, self.config, symbolic) for key, value in self.templates.items()},
            "placement": {key: value for key, value in self._placement(host).items() if key in self.fields},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "v1:sha256:" + hashlib.sha256(encoded).hexdigest()

    def resolve(self, unit_id, host, mounts):
        if self.operation != "capture" and not self.deferred:
            environment = self.captured.get(unit_id, {})
            if environment.get(ENV_INPUTS) != self._identity(host):
                raise ValueError("ColdSnap env template inputs or placement changed (or provenance is missing); recapture the recipe")
            if any(key not in environment or not isinstance(environment[key], str) for key in self.templates):
                raise ValueError("ColdSnap artifact is missing captured template values; recapture the recipe")
            return {key: environment[key] for key in self.templates}

        launch = self._placement(host)
        launch["runtime_cache_dir"] = CANONICAL_RUNTIME_CACHE_ROOT
        if "runtime_cache_dir" in self.fields:
            root = PurePosixPath(CANONICAL_RUNTIME_CACHE_ROOT)
            if any(root.is_relative_to(m.target) or PurePosixPath(m.target).is_relative_to(root) for m in mounts):
                raise ValueError("ColdSnap runtime cache is shadowed by a workload mount")
        if self.deferred:
            launch.update({name: "<unresolved:launch." + name + ">" for name in ("model_path", "model_revision")})
        elif self.fields & {"model_path", "model_revision"}:
            if host not in self.models:
                sctx = self.sctx or api.default_sctx()
                cache_dir = self.options.cache_dir or self.plan.cluster.cache_dir or str(sctx.config.hf_cache_dir)
                config = sctx.config.for_cluster(self.plan.cluster) if hasattr(sctx.config, "for_cluster") else sctx.config
                self.models[host] = probe_model_path(
                    host,
                    self.plan.recipe.model,
                    self.plan.recipe.model_revision,
                    cache_dir,
                    build_ssh_kwargs(config),
                    expected_revision=self.plan.recipe.model_revision,
                )
            model = self.models[host]
            launch["model_revision"] = model.revision
            if "model_path" in self.fields:
                volumes = {}
                for mount in mounts:
                    if mount.source in volumes and volumes[mount.source] != mount.target:
                        raise ValueError("ColdSnap model source is mounted at multiple destinations")
                    volumes[mount.source] = mount.target
                launch["model_path"] = workload_model_path(model.host_path, volumes, "docker")
        return {key: render_env_template(value, self.config, launch) for key, value in self.templates.items()}

    def stamp(self, units):
        for unit in units:
            environment = unit["environment"]
            if ENV_INPUTS in environment:
                raise ValueError(ENV_INPUTS + " is reserved for ColdSnap environment identity")
            if self.templates:
                environment[ENV_INPUTS] = self._identity(unit["host"])
            elif self.captured.get(unit["id"], {}).get(ENV_INPUTS):
                raise ValueError("ColdSnap env templates were removed or overridden; recapture the recipe")
