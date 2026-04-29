"""Tests for sparkrun.core.parallelism module."""

from __future__ import annotations

from sparkrun.core.parallelism import (
    PARALLELISM_KEYS,
    ParallelismConfig,
    extract_parallelism,
    extract_parallelism_meta,
)


class TestParallelismConfig:
    """Test ParallelismConfig dataclass."""

    def test_defaults(self):
        p = ParallelismConfig()
        assert p.tensor_parallel == 1
        assert p.pipeline_parallel == 1
        assert p.data_parallel == 1
        assert p.expert_parallel == 1
        assert p.context_parallel == 1

    def test_total_gpus_tp_only(self):
        p = ParallelismConfig(tensor_parallel=4)
        assert p.total_gpus == 4

    def test_total_gpus_pp_only(self):
        p = ParallelismConfig(pipeline_parallel=2)
        assert p.total_gpus == 2

    def test_total_gpus_tp_times_pp(self):
        p = ParallelismConfig(tensor_parallel=2, pipeline_parallel=3)
        assert p.total_gpus == 6

    def test_model_shard_factor_equals_total_gpus_when_dp_is_one(self):
        p = ParallelismConfig(tensor_parallel=2, pipeline_parallel=2)
        assert p.model_shard_factor == 4
        assert p.model_shard_factor == p.total_gpus  # dp=1 default

    def test_ep_cp_dont_affect_total_gpus(self):
        """Only tp/pp/dp count toward total GPU usage; ep/cp are intra-GPU concepts."""
        p = ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=1,
            expert_parallel=8,
            context_parallel=2,
        )
        assert p.total_gpus == 4  # tp * pp * dp, ep/cp irrelevant

    def test_total_nodes_defaults_to_one(self):
        p = ParallelismConfig()
        assert p.total_nodes == 1

    def test_total_nodes_dp_only(self):
        """DP=4 alone: 4 replicas, 1 GPU each → 4 nodes, 4 GPUs."""
        p = ParallelismConfig(data_parallel=4)
        assert p.total_nodes == 4
        assert p.total_gpus == 4
        assert p.model_shard_factor == 1  # each replica is a single-GPU model

    def test_total_nodes_hybrid_tp_dp(self):
        """TP=2, DP=2: 2 replicas of 2 GPUs each → 4 nodes, 4 GPUs, shard factor 2."""
        p = ParallelismConfig(tensor_parallel=2, data_parallel=2)
        assert p.total_nodes == 4
        assert p.total_gpus == 4
        assert p.model_shard_factor == 2

    def test_total_nodes_equals_tp_times_pp_times_dp(self):
        p = ParallelismConfig(tensor_parallel=2, pipeline_parallel=3, data_parallel=5)
        assert p.total_nodes == 30
        assert p.total_gpus == 30
        assert p.model_shard_factor == 6  # tp * pp — GPUs per replica


class TestExtractParallelism:
    """Test extract_parallelism() from various config sources."""

    def test_empty_dict(self):
        p = extract_parallelism({})
        assert p == ParallelismConfig()

    def test_dict_with_tp(self):
        p = extract_parallelism({"tensor_parallel": 4})
        assert p.tensor_parallel == 4
        assert p.pipeline_parallel == 1

    def test_dict_with_tp_and_pp(self):
        p = extract_parallelism({"tensor_parallel": 2, "pipeline_parallel": 3})
        assert p.tensor_parallel == 2
        assert p.pipeline_parallel == 3
        assert p.total_gpus == 6

    def test_string_values_coerced(self):
        p = extract_parallelism({"tensor_parallel": "4", "pipeline_parallel": "2"})
        assert p.tensor_parallel == 4
        assert p.pipeline_parallel == 2

    def test_none_values_ignored(self):
        p = extract_parallelism({"tensor_parallel": None, "pipeline_parallel": 2})
        assert p.tensor_parallel == 1
        assert p.pipeline_parallel == 2

    def test_all_keys(self):
        cfg = {
            "tensor_parallel": 2,
            "pipeline_parallel": 3,
            "data_parallel": 4,
            "expert_parallel": 5,
            "context_parallel": 6,
        }
        p = extract_parallelism(cfg)
        assert p.tensor_parallel == 2
        assert p.pipeline_parallel == 3
        assert p.data_parallel == 4
        assert p.expert_parallel == 5
        assert p.context_parallel == 6

    def test_extra_keys_ignored(self):
        p = extract_parallelism({"tensor_parallel": 2, "unrelated_key": 99})
        assert p.tensor_parallel == 2


class TestExtractParallelismMeta:
    """Test extract_parallelism_meta() for metadata dict generation."""

    def test_empty_dict(self):
        assert extract_parallelism_meta({}) == {}

    def test_default_values_omitted(self):
        assert extract_parallelism_meta({"tensor_parallel": 1}) == {}

    def test_non_default_values_included(self):
        meta = extract_parallelism_meta({"tensor_parallel": 2, "pipeline_parallel": 3})
        assert meta == {"tp": 2, "pp": 3}

    def test_uses_short_keys(self):
        meta = extract_parallelism_meta({"tensor_parallel": 4})
        assert "tp" in meta
        assert "tensor_parallel" not in meta

    def test_mixed_default_and_non_default(self):
        meta = extract_parallelism_meta(
            {
                "tensor_parallel": 2,
                "pipeline_parallel": 1,
                "data_parallel": 4,
            }
        )
        assert meta == {"tp": 2, "dp": 4}

    def test_none_values_omitted(self):
        meta = extract_parallelism_meta({"tensor_parallel": None, "pipeline_parallel": 2})
        assert meta == {"pp": 2}

    def test_string_values_coerced(self):
        meta = extract_parallelism_meta({"tensor_parallel": "2"})
        assert meta == {"tp": 2}


class TestParallelismKeys:
    """Test PARALLELISM_KEYS constant."""

    def test_has_all_five_dimensions(self):
        assert len(PARALLELISM_KEYS) == 5

    def test_long_short_pairs(self):
        longs = [k for k, _ in PARALLELISM_KEYS]
        shorts = [s for _, s in PARALLELISM_KEYS]
        assert "tensor_parallel" in longs
        assert "pipeline_parallel" in longs
        assert "tp" in shorts
        assert "pp" in shorts

    def test_keys_match_dataclass_fields(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ParallelismConfig)}
        for long_key, _ in PARALLELISM_KEYS:
            assert long_key in field_names
