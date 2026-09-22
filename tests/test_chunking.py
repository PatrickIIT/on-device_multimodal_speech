"""Unit tests for streaming audio chunking, buffering, and VAD."""

import numpy as np
import pytest

from streaming.chunking import AudioChunker, StreamBuffer
from streaming.latency_metrics import StreamingLatencyTracker
from streaming.streaming_inference import GenericStreamingASR
from streaming.vad import EnergyVAD


class TestAudioChunker:
    def test_chunking_exact_multiple(self):
        # 16000 Hz, 400 ms = 6400 samples per chunk
        # 2.0 seconds audio = 32000 samples -> exactly 5 chunks
        sr = 16000
        duration_s = 2.0
        audio = np.zeros(int(sr * duration_s), dtype=np.float32)
        chunker = AudioChunker(sample_rate=sr, chunk_size_ms=400)

        chunks = chunker.chunk_audio(audio)
        assert len(chunks) == 5
        for ch in chunks:
            assert len(ch) == 6400

    def test_chunking_with_padding(self):
        # 1.0 second audio at 400ms chunk (6400 samples):
        # 16000 samples -> 2 full chunks (12800) + 1 partial chunk (3200) -> 3 chunks
        sr = 16000
        audio = np.ones(16000, dtype=np.float32)
        chunker = AudioChunker(sample_rate=sr, chunk_size_ms=400)

        chunks = chunker.chunk_audio(audio, pad_final_chunk=True)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 6400
        # Check padding was filled with zeros
        assert chunks[-1][3200:].sum() == 0.0

    def test_chunking_empty_audio(self):
        chunker = AudioChunker(sample_rate=16000, chunk_size_ms=400)
        assert chunker.chunk_audio(np.array([], dtype=np.float32)) == []

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            AudioChunker(sample_rate=-16000)
        with pytest.raises(ValueError):
            AudioChunker(chunk_size_ms=0)
        with pytest.raises(ValueError):
            AudioChunker(lookahead_ms=-50)


class TestStreamBuffer:
    def test_buffer_push_and_pop(self):
        buffer = StreamBuffer(chunk_samples=100)
        assert not buffer.has_chunk()
        assert buffer.pop_chunk() is None

        # Push 60 samples
        buffer.push_samples(np.ones(60, dtype=np.float32))
        assert not buffer.has_chunk()

        # Push another 50 samples (total 110)
        buffer.push_samples(np.ones(50, dtype=np.float32))
        assert buffer.has_chunk()

        chunk = buffer.pop_chunk()
        assert chunk is not None
        assert len(chunk) == 100
        assert not buffer.has_chunk()

    def test_buffer_clear(self):
        buffer = StreamBuffer(chunk_samples=100)
        buffer.push_samples(np.ones(150, dtype=np.float32))
        buffer.clear()
        assert not buffer.has_chunk()
        assert buffer.pop_chunk() is None


class TestEnergyVAD:
    def test_silence_detection(self):
        vad = EnergyVAD(energy_threshold=0.01)
        silence = np.zeros(1600, dtype=np.float32)
        assert not vad.is_speech(silence)

    def test_speech_activation(self):
        vad = EnergyVAD(energy_threshold=0.01, min_speech_frames=2)
        speech_frame = np.ones(1600, dtype=np.float32) * 0.5

        # Frame 1: not active yet due to min_speech_frames=2
        res1 = vad.is_speech(speech_frame)
        assert not res1

        # Frame 2: transitions to active
        res2 = vad.is_speech(speech_frame)
        assert res2

        vad.reset()
        assert not vad.is_speech(np.zeros(1600, dtype=np.float32))


class TestStreamingASRInterface:
    def test_streaming_lifecycle(self):
        engine = GenericStreamingASR(chunk_size_ms=100)
        audio = np.zeros(3200, dtype=np.float32)  # 200 ms
        emitted = engine.accept_audio(audio)
        assert isinstance(emitted, list)
        assert engine.decode() == ""
        engine.reset()
        assert engine.decode() == ""


class TestStreamingLatencyTracker:
    def test_latency_summary(self):
        tracker = StreamingLatencyTracker()
        tracker.start_stream()
        tracker.record_chunk(chunk_duration_s=0.4, processing_time_s=0.08)
        tracker.record_chunk(chunk_duration_s=0.4, processing_time_s=0.07)
        tracker.mark_utterance_end()
        tracker.mark_final_token_emitted()

        summary = tracker.summary()
        assert summary["first_chunk_latency_ms"] == pytest.approx(80.0)
        assert summary["mean_cpt_ms"] == pytest.approx(75.0)
        assert summary["stream_rtf"] == pytest.approx(0.1875)
