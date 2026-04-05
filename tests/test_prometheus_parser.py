"""Tests for Prometheus text format parser and metric-to-sample adapter."""

from __future__ import annotations

from sparkrun.core.prometheus import extract_label, parse_prometheus_text
from sparkrun.core.monitoring import prometheus_to_sample


# ---------------------------------------------------------------------------
# Sample nv-monitor output
# ---------------------------------------------------------------------------

SAMPLE_PROMETHEUS_OUTPUT = """\
# HELP nv_cpu_usage_percent CPU usage percentage
# TYPE nv_cpu_usage_percent gauge
nv_cpu_usage_percent{cpu="overall"} 42.5
nv_cpu_usage_percent{cpu="0"} 80.0
nv_cpu_usage_percent{cpu="1"} 5.0
# HELP nv_cpu_temperature_celsius CPU temperature
# TYPE nv_cpu_temperature_celsius gauge
nv_cpu_temperature_celsius 55
# HELP nv_cpu_frequency_mhz CPU frequency
# TYPE nv_cpu_frequency_mhz gauge
nv_cpu_frequency_mhz 3200
# HELP nv_memory_total_bytes Total system memory
# TYPE nv_memory_total_bytes gauge
nv_memory_total_bytes 137438953472
# HELP nv_memory_used_bytes Used system memory
# TYPE nv_memory_used_bytes gauge
nv_memory_used_bytes 68719476736
# HELP nv_memory_bufcache_bytes Buffers/cache memory
# TYPE nv_memory_bufcache_bytes gauge
nv_memory_bufcache_bytes 10737418240
# HELP nv_gpu_utilization_percent GPU utilization
# TYPE nv_gpu_utilization_percent gauge
nv_gpu_utilization_percent{gpu="0"} 95.0
# HELP nv_gpu_temperature_celsius GPU temperature
# TYPE nv_gpu_temperature_celsius gauge
nv_gpu_temperature_celsius{gpu="0"} 72
# HELP nv_gpu_power_watts GPU power draw
# TYPE nv_gpu_power_watts gauge
nv_gpu_power_watts{gpu="0"} 285.5
# HELP nv_gpu_power_limit_watts GPU power limit
# TYPE nv_gpu_power_limit_watts gauge
nv_gpu_power_limit_watts{gpu="0"} 350
# HELP nv_gpu_clock_mhz GPU clock speed
# TYPE nv_gpu_clock_mhz gauge
nv_gpu_clock_mhz{gpu="0",type="graphics"} 1800
nv_gpu_clock_mhz{gpu="0",type="memory"} 1200
# HELP nv_gpu_memory_total_bytes GPU memory total
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu="0"} 137438953472
# HELP nv_gpu_memory_used_bytes GPU memory used
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu="0"} 120259084288
# HELP nv_gpu_encoder_utilization_percent GPU encoder utilization
# TYPE nv_gpu_encoder_utilization_percent gauge
nv_gpu_encoder_utilization_percent{gpu="0"} 15
# HELP nv_gpu_decoder_utilization_percent GPU decoder utilization
# TYPE nv_gpu_decoder_utilization_percent gauge
nv_gpu_decoder_utilization_percent{gpu="0"} 8
# HELP nv_gpu_fan_speed_percent GPU fan speed
# TYPE nv_gpu_fan_speed_percent gauge
nv_gpu_fan_speed_percent{gpu="0"} 65
# HELP nv_gpu_info GPU information
# TYPE nv_gpu_info gauge
nv_gpu_info{gpu="0",name="GH200 120GB"} 1
# HELP nv_load_average System load average
# TYPE nv_load_average gauge
nv_load_average{interval="1m"} 2.5
nv_load_average{interval="5m"} 1.8
nv_load_average{interval="15m"} 1.2
# HELP nv_system_uptime_seconds System uptime
# TYPE nv_system_uptime_seconds gauge
nv_system_uptime_seconds 86400
# HELP nv_swap_total_bytes Total swap
# TYPE nv_swap_total_bytes gauge
nv_swap_total_bytes 0
# HELP nv_swap_used_bytes Used swap
# TYPE nv_swap_used_bytes gauge
nv_swap_used_bytes 0
"""


class TestParsePrometheusText:
    def test_basic_parsing(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_OUTPUT)
        assert metrics['nv_cpu_usage_percent{cpu="overall"}'] == 42.5
        assert metrics['nv_gpu_utilization_percent{gpu="0"}'] == 95.0
        assert metrics["nv_cpu_temperature_celsius"] == 55.0

    def test_empty_input(self):
        assert parse_prometheus_text("") == {}

    def test_comments_only(self):
        text = "# HELP foo\n# TYPE foo gauge\n"
        assert parse_prometheus_text(text) == {}

    def test_malformed_lines_skipped(self):
        text = "good_metric 42\nbad line no value\nanother_good 99\n"
        metrics = parse_prometheus_text(text)
        assert metrics["good_metric"] == 42.0
        assert metrics["another_good"] == 99.0
        assert len(metrics) == 2

    def test_metric_with_labels(self):
        text = 'my_metric{label1="a",label2="b"} 123.45\n'
        metrics = parse_prometheus_text(text)
        assert metrics['my_metric{label1="a",label2="b"}'] == 123.45

    def test_metric_without_labels(self):
        text = "simple_metric 99.9\n"
        metrics = parse_prometheus_text(text)
        assert metrics["simple_metric"] == 99.9

    def test_special_values(self):
        # NaN is supported by the regex alternation
        text = "nan_metric NaN\n"
        metrics = parse_prometheus_text(text)
        import math

        assert math.isnan(metrics["nan_metric"])

    def test_plain_inf(self):
        # Plain "Inf" (without sign prefix) is matched by the regex alternation
        text = "inf_metric Inf\n"
        metrics = parse_prometheus_text(text)
        assert metrics["inf_metric"] == float("inf")

    def test_scientific_notation(self):
        text = "sci_metric 1.5e3\n"
        metrics = parse_prometheus_text(text)
        assert metrics["sci_metric"] == 1500.0


class TestExtractLabel:
    def test_extract_existing_label(self):
        key = 'nv_gpu_info{gpu="0",name="GH200 120GB"}'
        assert extract_label(key, "name") == "GH200 120GB"
        assert extract_label(key, "gpu") == "0"

    def test_extract_missing_label(self):
        key = 'nv_gpu_info{gpu="0"}'
        assert extract_label(key, "name") is None

    def test_no_labels(self):
        assert extract_label("nv_cpu_temp", "gpu") is None


class TestPrometheusToSample:
    def test_full_conversion(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_OUTPUT)
        sample = prometheus_to_sample(metrics, "testhost")

        assert sample.hostname == "testhost"
        assert sample.cpu_usage_pct == "42.5"
        assert sample.cpu_temp_c == "55"
        assert sample.gpu_util_pct == "95"
        assert sample.gpu_name == "GH200 120GB"
        assert sample.gpu_encoder_pct == "15"
        assert sample.gpu_decoder_pct == "8"
        assert sample.gpu_fan_pct == "65"
        # Memory: 137438953472 bytes = 131072 MB
        assert sample.mem_total_mb == "131072"
        # Memory used: 68719476736 = 65536 MB
        assert sample.mem_used_mb == "65536"
        assert sample.mem_bufcache_mb == "10240"
        assert sample.gpu_power_w == "285.5"
        assert sample.cpu_load_1m == "2.5"

    def test_empty_metrics(self):
        sample = prometheus_to_sample({}, "emptyhost")
        assert sample.hostname == "emptyhost"
        assert sample.cpu_usage_pct == ""
        assert sample.gpu_util_pct == ""
        assert sample.gpu_name == ""

    def test_partial_metrics(self):
        metrics = {
            'nv_cpu_usage_percent{cpu="overall"}': 50.0,
            'nv_gpu_utilization_percent{gpu="0"}': 75.0,
        }
        sample = prometheus_to_sample(metrics, "partial")
        assert sample.cpu_usage_pct == "50"
        assert sample.gpu_util_pct == "75"
        assert sample.mem_total_mb == ""
        assert sample.gpu_name == ""
