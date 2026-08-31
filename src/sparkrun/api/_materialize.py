"""Read-only materialization of a :class:`RunPlan` into launch units."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from sparkrun.api._context import default_sctx
from sparkrun.api._models import (
    ResolvedAdapterTopology,
    ResolvedExecutionGraph,
    ResolvedLaunchSpec,
    ResolvedLaunchUnit,
    ResolvedMount,
    ResolvedProcessGroup,
    ResolvedServiceDomain,
    ResolvedWorker,
    RunOptions,
    RunPlan,
)


def materialize(
    options: RunOptions,
    *,
    plan: RunPlan | None = None,
    comm_env=None,
    sctx=None,
    images_by_node: Sequence[str] | None = None,
) -> ResolvedLaunchSpec:
    """Resolve the launch data an integration needs without starting it.

    Placement is reused from *plan* when provided. The function deliberately
    avoids image pulls, model distribution, cache creation, and network
    probing. A caller which already resolved a cluster communication
    environment may pass *comm_env* to materialize its per-host settings.
    A shared image-preparation caller may pass *images_by_node* to replace the
    declarative recipe image plan with already-resident, positionally aligned
    references.
    Integrations that need immutable process identity should require recipe
    images in ``name@sha256:...`` form.
    """

    if sctx is None:
        sctx = default_sctx()
    if plan is None:
        from sparkrun.api._run import plan as build_plan

        plan = build_plan(options, sctx=sctx)

    recipe = plan.recipe
    runtime = plan.runtime
    spec_engine = runtime.get_family()
    hosts = list(plan.host_list)
    if not hosts:
        raise ValueError("cannot materialize a launch with no selected hosts")
    if not plan.is_solo and runtime.cluster_strategy() != "native":
        raise ValueError("materialized launch units require a native distributed runtime; got %s" % runtime.runtime_name)

    overrides = dict(options.overrides)
    config_chain = recipe.build_config_chain(overrides)
    from sparkrun.core.parallelism import extract_parallelism

    parallelism = extract_parallelism(config_chain)
    placement = plan.placement
    implicit_slots: tuple[tuple[str, int], ...] = ()
    if placement is not None:
        world_size = placement.total_ranks
    elif bool(options.solo) or recipe.mode == "solo":
        world_size = 1
        implicit_slots = ((hosts[0], 0),)
    else:
        world_size = runtime.world_size(parallelism, recipe=recipe, cluster=plan.cluster)
        if world_size == len(hosts):
            implicit_slots = tuple((host, 0) for host in hosts)
        elif len(hosts) == 1 and plan.cluster.hardware_for(hosts[0]).total_gpus >= world_size:
            implicit_slots = tuple((hosts[0], gpu) for gpu in range(world_size))
        else:
            raise ValueError(
                "materialized launch requires an explicit worker placement for %d workers across %d hosts" % (world_size, len(hosts))
            )
    if world_size <= 0:
        raise ValueError("native materialization requires at least one worker")

    from sparkrun.core.images import ImagePlan, resolve_image_plan

    default_image = runtime.resolve_container(recipe, overrides)
    image_plan = resolve_image_plan(
        recipe,
        default_image,
        hosts,
        cluster_hosts=list(plan.cluster.hosts),
    )
    if images_by_node is not None:
        prepared = tuple(str(image).strip() for image in images_by_node)
        if len(prepared) != len(hosts):
            raise ValueError("materialized image override has %d image(s) for %d host(s)" % (len(prepared), len(hosts)))
        if not all(prepared):
            raise ValueError("materialized image override contains an empty reference")
        image_plan = ImagePlan(default_image=prepared[0], images_by_node=prepared)

    cache_dir = options.cache_dir or getattr(plan.cluster, "cache_dir", None) or str(sctx.config.hf_cache_dir)
    from sparkrun.orchestration.primitives import build_volumes, resolved_model_volume

    model_volumes = resolved_model_volume(recipe)
    volumes = build_volumes(
        cache_dir,
        extra={**runtime.get_extra_volumes(), **model_volumes},
    )
    # An execution strategy may keep a privileged controller inside the
    # workload container. The pinned model inputs are prepared before that
    # container starts and must remain immutable, so do not let controller root
    # leak ownership or mutations back into sparkrun's shared model cache.
    read_only_targets = {"/cache/huggingface", *(str(target) for target in model_volumes.values())}
    mounts_by_target = {
        str(target): ResolvedMount(
            source=str(source),
            target=str(target),
            read_only=str(target) in read_only_targets,
        )
        for source, target in volumes.items()
    }
    for mount in _resolved_executor_mounts(options, plan=plan, runtime=runtime, sctx=sctx):
        mounts_by_target[mount.target] = mount
    mounts = tuple(mounts_by_target[target] for target in sorted(mounts_by_target))

    from sparkrun.core.launcher import resolve_platform_env_defaults
    from sparkrun.utils import merge_env

    head_hardware = plan.cluster.hardware_for(hosts[0])
    platform_env = resolve_platform_env_defaults(runtime, head_hardware)
    cluster_env = plan.cluster.resolve_env() if getattr(plan.cluster, "env", None) else {}
    declared_env = {**platform_env, **cluster_env, **(recipe.env or {})}
    runtime_env = runtime.get_solo_env() if plan.is_solo else runtime.get_cluster_env(head_ip=hosts[0], num_nodes=len(hosts))
    environment = merge_env(
        runtime.get_common_env(),
        runtime_env,
        declared_env,
        runtime.get_extra_env(),
    )

    assignments: list[tuple[int, str, int, int]] = []
    replica_size = parallelism.tensor_parallel * parallelism.pipeline_parallel
    standard_vllm_layout = spec_engine == "vllm" and replica_size > 0 and world_size == replica_size * parallelism.data_parallel
    claimed_devices: set[tuple[str, int]] = set()
    for worker_rank in range(world_size):
        host, gpu = (
            (placement.host_for_rank(worker_rank), placement.local_gpu_for_rank(worker_rank))
            if placement is not None
            else implicit_slots[worker_rank]
        )
        if (host, gpu) in claimed_devices:
            raise ValueError("materialized launches do not support workers sharing accelerator %s on %s" % (gpu, host))
        claimed_devices.add((host, gpu))
        service_index = worker_rank // replica_size if standard_vllm_layout else 0
        assignments.append((worker_rank, host, gpu, service_index))

    # A launch unit may own several local workers, but never crosses a service
    # boundary. This permits multiple DP/PD service containers on one host.
    unit_keys: list[tuple[int, str]] = []
    workers_by_unit: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for worker_rank, host, gpu, service_index in assignments:
        key = (service_index, host)
        if key not in workers_by_unit:
            unit_keys.append(key)
            workers_by_unit[key] = []
        workers_by_unit[key].append((worker_rank, gpu))

    service_units: dict[int, list[tuple[int, str]]] = {}
    for key in unit_keys:
        service_units.setdefault(key[0], []).append(key)

    # A vLLM distributed service uses one homogeneous local process count for
    # every participating launch unit. Keep the execution-graph schema more
    # general, but reject layouts this adapter cannot launch faithfully.
    if spec_engine == "vllm":
        for service_index, keys in service_units.items():
            local_worker_counts = {len(workers_by_unit[key]) for key in keys}
            if len(local_worker_counts) != 1:
                raise ValueError("vLLM service %d requires the same number of local workers on every launch unit" % service_index)

    units: list[ResolvedLaunchUnit] = []
    workers: list[ResolvedWorker] = []
    worker_ids_by_service: dict[int, list[str]] = {}
    for unit_index, key in enumerate(unit_keys):
        service_index, host = key
        unit_id = "unit-%d" % unit_index
        service_id = "service-%d" % service_index
        unit_workers = workers_by_unit[key]
        devices = tuple(str(gpu) for _rank, gpu in unit_workers)
        host_index = hosts.index(host)
        image = image_plan.image_for_node(host_index)
        if plan.is_solo:
            command_text = runtime.generate_command(
                recipe=recipe,
                overrides=overrides,
                is_cluster=False,
                num_nodes=1,
                head_ip=host,
            )
        else:
            worker_ranks = tuple(rank for rank, _gpu in unit_workers)
            if len(worker_ranks) == 1:
                command_text = runtime.generate_node_command(
                    recipe=recipe,
                    overrides=overrides,
                    head_ip=hosts[0],
                    num_nodes=len(hosts),
                    node_rank=worker_ranks[0],
                    init_port=options.init_port,
                    hosts=hosts,
                    placement=placement,
                )
            else:
                generator = getattr(runtime, "generate_launch_unit_command", None)
                if not callable(generator):
                    raise ValueError(
                        "%s cannot materialize %d local workers in one launch unit" % (runtime.runtime_name, len(worker_ranks))
                    )
                keys = service_units[service_index]
                command_text = generator(
                    recipe=recipe,
                    overrides=overrides,
                    head_ip=hosts[0],
                    init_port=options.init_port,
                    worker_ranks=worker_ranks,
                    service_index=service_index,
                    service_unit_index=keys.index(key),
                    service_unit_count=len(keys),
                    service_hosts=[service_host for _service, service_host in keys],
                )
        unit_environment = dict(environment)
        if comm_env:
            host_environment = comm_env.get_env(host)
            host_environment = runtime.finalize_host_comm_env(host_environment)
            unit_environment.update(host_environment)
        units.append(
            ResolvedLaunchUnit(
                id=unit_id,
                index=unit_index,
                host=host,
                devices=devices,
                image=image,
                image_digest=_digest_from_image(image),
                # Sparkrun executes generated serve commands as Bash scripts,
                # not as a tokenized argv. Preserve pipes, redirects, variable
                # expansion, and compound commands across the adapter boundary.
                command=("bash", "--noprofile", "--norc", "-c", command_text),
                environment=unit_environment,
                mounts=mounts,
            )
        )
        for process_slot, (worker_rank, _gpu) in enumerate(unit_workers):
            worker_id = "worker-%d" % worker_rank
            workers.append(
                ResolvedWorker(
                    id=worker_id,
                    unit=unit_id,
                    service=service_id,
                    process_slot=process_slot,
                    device_slots=(process_slot,),
                )
            )
            worker_ids_by_service.setdefault(service_index, []).append(worker_id)

    services = tuple(
        ResolvedServiceDomain(id="service-%d" % index, role="%s:serve" % spec_engine, workers=tuple(worker_ids))
        for index, worker_ids in sorted(worker_ids_by_service.items())
    )
    groups = tuple(
        ResolvedProcessGroup(
            id="group-%d" % index,
            kind="%s:world" % spec_engine,
            service="service-%d" % index,
            members=tuple(worker_ids),
        )
        for index, worker_ids in sorted(worker_ids_by_service.items())
    )
    adapter_payload = {
        "runtime": runtime.runtime_name,
        "dimensions": {
            "tensor": parallelism.tensor_parallel,
            "pipeline": parallelism.pipeline_parallel,
            "data": parallelism.data_parallel,
            "expert": parallelism.expert_parallel,
            "context": parallelism.context_parallel,
        },
    }
    canonical = json.dumps(adapter_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    execution = ResolvedExecutionGraph(
        workers=tuple(sorted(workers, key=lambda worker: int(worker.id.removeprefix("worker-")))),
        groups=groups,
        services=services,
        adapter=ResolvedAdapterTopology(
            schema="%s:sparkrun-v1" % spec_engine,
            digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
            payload=adapter_payload,
        ),
    )

    return ResolvedLaunchSpec(
        format=2,
        kind="sparkrun-resolved-launch",
        recipe=recipe.qualified_name,
        cluster_id=plan.cluster_id,
        runtime=runtime.runtime_name,
        engine=spec_engine,
        model=recipe.model,
        model_revision=recipe.model_revision or "",
        world_size=world_size,
        tensor_parallel=parallelism.tensor_parallel,
        node_count=len(hosts),
        cache_dir=cache_dir,
        units=tuple(units),
        execution=execution,
    )


def _resolved_executor_mounts(options: RunOptions, *, plan: RunPlan, runtime, sctx) -> tuple[ResolvedMount, ...]:
    """Return recipe/CLI executor mounts for execution-strategy consumers.

    Normal launches hand ``ExecutorConfig.volumes`` directly to the selected
    executor.  Materialized launch units must carry the same mounts because an
    execution strategy creates the containers itself, so anything the executor
    would have added is otherwise simply absent.  Explicit executor mounts win
    when they target a standard runtime mount.
    """

    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.utils.shell import assert_safe_mount_source

    cli_overrides: dict[str, object] = {}
    if options.executor:
        cli_overrides["executor"] = options.executor
    if options.executor_config:
        cli_overrides.update(options.executor_config)
    hosts = list(plan.host_list)
    executor = resolve_executor(
        recipe=plan.recipe,
        cluster=plan.cluster,
        runtime=runtime,
        config=sctx.config,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        host_hardware=plan.cluster.hardware_for(hosts[0]),
        v=getattr(sctx, "variables", None),
    )
    mounts: list[ResolvedMount] = []
    for raw in executor.config.volumes or ():
        value = str(raw).strip()
        fields = value.split(":")
        if len(fields) == 1:
            source = target = fields[0]
            read_only = False
        elif len(fields) == 2:
            source, target = fields
            read_only = False
        elif len(fields) == 3 and fields[2] in {"ro", "rw"}:
            source, target, mode = fields
            read_only = mode == "ro"
        else:
            raise ValueError(f"executor volume must be PATH, SOURCE:TARGET, or SOURCE:TARGET:{{ro,rw}}: {value}")
        if not source or not target:
            raise ValueError(f"executor volume source and target must be non-empty: {value}")
        assert_safe_mount_source(source)
        mounts.append(ResolvedMount(source=source, target=target, read_only=read_only))
    return tuple(mounts)


def _digest_from_image(image: str) -> str:
    marker = "@sha256:"
    if marker in image:
        digest = "sha256:" + image.rsplit(marker, 1)[1]
        if len(digest) == 71 and all(c in "0123456789abcdef" for c in digest[7:]):
            return digest
    if image.startswith("sha256:") and len(image) == 71:
        return image
    return ""


__all__ = ["materialize"]
