from miles.utils.ft.controller.mini_prometheus.scraper import parse_prometheus_text


class TestParsePrometheusText:
    def test_simple_metric(self) -> None:
        text = "gpu_temperature_celsius 75.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temperature_celsius"
        assert samples[0].value == 75.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = 'gpu_temperature_celsius{gpu="0",node="n1"} 82.5\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].labels == {"gpu": "0", "node": "n1"}
        assert samples[0].value == 82.5

    def test_skips_comments_and_help(self) -> None:
        text = (
            "# HELP gpu_temp GPU temperature\n"
            "# TYPE gpu_temp gauge\n"
            "gpu_temp 42.0\n"
        )
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temp"

    def test_multiple_metrics(self) -> None:
        text = (
            "metric_a 1.0\n"
            "metric_b 2.0\n"
            "metric_c{label=\"x\"} 3.0\n"
        )
        samples = parse_prometheus_text(text)
        assert len(samples) == 3

    def test_metric_with_timestamp(self) -> None:
        text = "http_requests_total 1000 1700000000000\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == 1000.0

    def test_empty_input_returns_empty(self) -> None:
        assert parse_prometheus_text("") == []

    def test_only_comments_returns_empty(self) -> None:
        text = "# HELP metric_a A metric\n# TYPE metric_a gauge\n"
        assert parse_prometheus_text(text) == []

    def test_malformed_value_is_skipped(self) -> None:
        text = "metric_a not_a_number\nmetric_b 2.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "metric_b"

    def test_scientific_notation_value(self) -> None:
        text = "metric_a 1.23e4\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == 12300.0

    def test_blank_lines_skipped(self) -> None:
        text = "\n\nmetric_a 1.0\n\nmetric_b 2.0\n\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 2

    def test_colon_in_metric_name(self) -> None:
        text = 'job:request_latency_seconds:mean5m{job="api"} 0.42\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "job:request_latency_seconds:mean5m"
        assert samples[0].labels == {"job": "api"}

    def test_positive_infinity(self) -> None:
        text = "metric_a +Inf\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == float("inf")

    def test_negative_infinity(self) -> None:
        text = "metric_a -Inf\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == float("-inf")

    def test_nan_value(self) -> None:
        import math

        text = "metric_a NaN\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert math.isnan(samples[0].value)

    def test_negative_value(self) -> None:
        text = "metric_a -1.5\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == -1.5
