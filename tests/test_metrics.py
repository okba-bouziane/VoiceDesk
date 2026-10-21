from __future__ import annotations

from livekit.agents import metrics as lkm

from agent.metrics import TurnTracker


def test_turn_is_complete_once_eou_llm_and_tts_arrive():
    tr = TurnTracker("v-test")
    assert (
        tr.feed(
            lkm.EOUMetrics(
                timestamp=0,
                end_of_utterance_delay=0.6,
                transcription_delay=0.3,
                on_user_turn_completed_delay=0.0,
                speech_id="s1",
            )
        )
        is None
    )
    assert (
        tr.feed(
            lkm.LLMMetrics(
                label="x",
                request_id="r",
                timestamp=0,
                duration=1,
                ttft=0.8,
                cancelled=False,
                completion_tokens=5,
                prompt_tokens=5,
                prompt_cached_tokens=0,
                total_tokens=10,
                tokens_per_second=5,
                speech_id="s1",
            )
        )
        is None
    )
    turn = tr.feed(
        lkm.TTSMetrics(
            label="x",
            request_id="r",
            timestamp=0,
            ttfb=0.4,
            duration=1,
            audio_duration=2,
            cancelled=False,
            characters_count=10,
            streamed=False,
            speech_id="s1",
        )
    )
    assert turn is not None
    assert round(turn.end_to_end, 2) == 1.8
    assert turn.as_dict()["stt_delay"] == 0.3


def test_cancelled_generations_do_not_count():
    tr = TurnTracker("v-test")
    assert (
        tr.feed(
            lkm.LLMMetrics(
                label="x",
                request_id="r",
                timestamp=0,
                duration=1,
                ttft=-1,
                cancelled=True,
                completion_tokens=0,
                prompt_tokens=5,
                prompt_cached_tokens=0,
                total_tokens=5,
                tokens_per_second=0,
                speech_id="s2",
            )
        )
        is None
    )
