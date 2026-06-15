"""Tests for spark-vllm-docker .env -> cluster mapping (sparkrun.core.svd_import)."""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterError
from sparkrun.core.svd_import import build_svd_import


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_build_maps_nodes_eth_and_container(tmp_path):
    env = _write(
        tmp_path,
        "cluster.env",
        'CLUSTER_NODES=192.168.11.14,192.168.11.16\nETH_IF=enp1s0f1np1\nCONTAINER_HF_TOKEN="hf_secret"\nCONTAINER_NCCL_DEBUG=INFO\n',
    )
    imp = build_svd_import(str(env))

    assert imp.hosts == ["192.168.11.14", "192.168.11.16"]
    assert imp.fabric_interfaces == ["*np1"]  # port extracted from ETH_IF
    assert imp.env == {"HF_TOKEN": "${CONTAINER_HF_TOKEN}", "NCCL_DEBUG": "${CONTAINER_NCCL_DEBUG}"}
    assert imp.env_file == str(env.resolve())
    assert imp.sync_source == "spark_vllm_docker:%s" % env.resolve()
    assert imp.default_name == "cluster"  # file stem


def test_build_drops_derived_and_run_level(tmp_path):
    env = _write(
        tmp_path,
        "cluster.env",
        "CLUSTER_NODES=a,b\n"
        "COPY_HOSTS=b\n"
        "LOCAL_IP=a\n"
        "MASTER_PORT=29501\n"
        "IB_IF=rocep1s0f1,roceP2p1s0f1\n"
        "CONTAINER_NAME=vllm_node\n"
        "CONTAINER_NCCL_NET_PLUGIN=none\n"
        "CONTAINER_NCCL_IB_MERGE_NICS=0\n"
        "CONTAINER_HF_TOKEN=x\n",
    )
    imp = build_svd_import(str(env))

    # Derived NCCL mesh vars and CONTAINER_NAME never carried.
    assert imp.env == {"HF_TOKEN": "${CONTAINER_HF_TOKEN}"}
    dropped_blob = " ".join(imp.dropped)
    for key in (
        "CONTAINER_NCCL_NET_PLUGIN",
        "CONTAINER_NCCL_IB_MERGE_NICS",
        "CONTAINER_NAME",
        "COPY_HOSTS",
        "LOCAL_IP",
        "MASTER_PORT",
        "IB_IF",
    ):
        assert key in dropped_blob


def test_default_name_falls_back_to_parent_dir_for_dotenv(tmp_path):
    sub = tmp_path / "spark-vllm-docker"
    sub.mkdir()
    env = _write(sub, ".env", "CLUSTER_NODES=a\n")
    imp = build_svd_import(str(env))
    assert imp.default_name == "spark-vllm-docker"


def test_eth_if_without_np_suffix_uses_exact_name(tmp_path):
    env = _write(tmp_path, "c.env", "CLUSTER_NODES=a\nETH_IF=eth0\n")
    imp = build_svd_import(str(env))
    assert imp.fabric_interfaces == ["eth0"]


def test_no_cluster_nodes_errors(tmp_path):
    env = _write(tmp_path, "c.env", "ETH_IF=enp1s0f1np1\n")
    with pytest.raises(ClusterError, match="CLUSTER_NODES"):
        build_svd_import(str(env))


def test_missing_file_errors(tmp_path):
    with pytest.raises(ClusterError, match="could not read"):
        build_svd_import(str(tmp_path / "nope.env"))
