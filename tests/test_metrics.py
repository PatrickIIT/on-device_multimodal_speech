"""Unit tests for speech recognition and efficiency metrics."""

import math
import pytest
from benchmarks.metrics import (
    cer,
    compression_ratio,
    latency_statistics,
    real_time_factor,
    wer,
)


class TestWordErrorRate:
    def test_wer_exact_match(self):
        ref = "habari ya asubuhi rafiki"
        hyp = "habari ya asubuhi rafiki"
        assert wer(ref, hyp) == 0.0

    def test_wer_substitution(self):
        ref = "habari ya asubuhi rafiki"
        hyp = "habari ya jioni rafiki"
        # 1 substitution out of 4 words = 0.25
        assert wer(ref, hyp) == pytest.approx(0.25)

    def test_wer_deletion(self):
        ref = "habari ya asubuhi rafiki"
        hyp = "habari asubuhi rafiki"
        # 1 deletion out of 4 words = 0.25
        assert wer(ref, hyp) == pytest.approx(0.25)

    def test_wer_insertion(self):
        ref = "habari ya asubuhi rafiki"
        hyp = "habari kubwa ya asubuhi rafiki"
        # 1 insertion out of 4 words = 0.25
        assert wer(ref, hyp) == pytest.approx(0.25)

    def test_wer_empty_strings(self):
        assert wer("", "") == 0.0
        assert wer("habari", "") == 1.0
        assert wer("", "habari") == 1.0

    def test_wer_multiple_utterances(self):
        refs = ["habari ya asubuhi", "jina langu ni patrick"]
        hyps = ["habari ya jioni", "jina langu ni patrick"]
        # 1 error in 3 words + 0 errors in 4 words = 1/7
        assert wer(refs, hyps) == pytest.approx(1.0 / 7.0)

    def test_wer_mismatched_lengths(self):
        with pytest.raises(ValueError):
            wer(["habari"], ["habari", "jioni"])


class TestCharacterErrorRate:
    def test_cer_exact_match(self):
        ref = "kiswahili"
        hyp = "kiswahili"
        assert cer(ref, hyp) == 0.0

    def test_cer_substitution(self):
        ref = "jambo"
        hyp = "jambo!"
        # 1 insertion in 5 chars = 0.2
        assert cer(ref, hyp) == pytest.approx(0.2)

    def test_cer_multiple(self):
        refs = ["habari", "mambo"]
        hyps = ["habari", "mambp"]
        # 0 in 6 + 1 in 5 = 1/11
        assert cer(refs, hyps) == pytest.approx(1.0 / 11.0)


class TestRealTimeFactor:
    def test_rtf_faster_than_realtime(self):
        # 10s audio processed in 2s -> RTF = 0.2
        assert real_time_factor(2.0, 10.0) == pytest.approx(0.2)

    def test_rtf_slower_than_realtime(self):
        # 5s audio processed in 10s -> RTF = 2.0
        assert real_time_factor(10.0, 5.0) == pytest.approx(2.0)

    def test_rtf_invalid_inputs(self):
        with pytest.raises(ValueError):
            real_time_factor(-1.0, 10.0)
        with pytest.raises(ValueError):
            real_time_factor(1.0, 0.0)


class TestCompressionRatio:
    def test_compression_ratio(self):
        # 100 MB reduced to 25 MB = 4.0x
        assert compression_ratio(100.0, 25.0) == pytest.approx(4.0)

    def test_compression_ratio_invalid(self):
        with pytest.raises(ValueError):
            compression_ratio(0.0, 25.0)
        with pytest.raises(ValueError):
            compression_ratio(100.0, -5.0)


class TestLatencyStatistics:
    def test_latency_statistics_calculation(self):
        # Synthetic latencies: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = latency_statistics(latencies)

        assert stats["count"] == 10
        assert stats["mean"] == pytest.approx(55.0)
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(100.0)
        assert stats["p50"] == pytest.approx(55.0)
        assert stats["p90"] == pytest.approx(91.0)
        assert stats["p95"] == pytest.approx(95.5)

    def test_latency_statistics_empty(self):
        with pytest.raises(ValueError):
            latency_statistics([])
