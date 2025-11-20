# app/vad.py
import wave, contextlib, struct, os
from typing import List, Tuple

def _read_pcm16_mono(wav_path: str):
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        assert wf.getnchannels() == 1, "need mono"
        assert wf.getsampwidth() == 2, "need 16-bit"
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return sample_rate, pcm

def _frame_bytes(pcm: bytes, frame_len_samples: int):
    step = frame_len_samples * 2  # 16-bit mono
    for i in range(0, len(pcm), step):
        yield pcm[i:i+step]

def _bytes_to_duration(num_bytes: int, sample_rate: int):
    samples = num_bytes // 2
    return samples / float(sample_rate)

def compute_vad_silences(
    wav_16k_mono_path: str,
    aggressiveness: int = 3,
    frame_ms: int = 30,
    hangover_ms: int = 300,
) -> List[Tuple[float, float]]:
    """
    webrtcvad 기반 침묵(silence) 구간 리스트 반환.
    - aggressiveness: 0(관대)~3(공격적). 3 권장(잡음/배경음에 강함)
    - frame_ms: 10|20|30 권장
    - hangover_ms: 말 끝나고 추가로 붙여주는 비활성 유지 시간
    """
    try:
        import webrtcvad
    except Exception:
        # webrtcvad 미설치 시 빈 리스트 (후속 로직은 STT 차이로 fallback)
        return []

    assert frame_ms in (10, 20, 30), "frame_ms must be 10/20/30"
    sr, pcm = _read_pcm16_mono(wav_16k_mono_path)
    assert sr == 16000, "need 16k mono for VAD"

    vad = webrtcvad.Vad(aggressiveness)
    frame_len_samples = int(sr * frame_ms / 1000.0)
    frames = list(_frame_bytes(pcm, frame_len_samples))
    ts = 0.0
    speech_flags = []
    for f in frames:
        if len(f) < frame_len_samples*2:  # 마지막 짜투리
            break
        is_speech = vad.is_speech(f, sr)
        speech_flags.append((ts, ts + frame_ms/1000.0, is_speech))
        ts += frame_ms/1000.0

    # hangover(말 끝난 직후 약간 더 말로 간주) 제거 → 침묵 계산에 넣지 않음
    last_speech_end = -1e9
    speech_ranges = []
    cur_state = False
    start_t = 0.0
    for s, e, flag in speech_flags:
        if flag and not cur_state:
            cur_state = True; start_t = s
        elif (not flag) and cur_state:
            cur_state = False; last_speech_end = e
            speech_ranges.append((start_t, e))
    if cur_state:
        speech_ranges.append((start_t, speech_flags[-1][1]))

    # silence = 전체에서 speech를 뺀 보Complement
    silences: List[Tuple[float,float]] = []
    cur = 0.0
    for (s,e) in speech_ranges:
        if s > cur:
            silences.append((cur, s))
        cur = e
    if speech_flags:
        end_total = speech_flags[-1][1]
        if end_total > cur:
            silences.append((cur, end_total))
    return [(float(a), float(b)) for a,b in silences]

def sum_silence_between(silences: List[Tuple[float,float]], a: float, b: float) -> float:
    if not silences or b <= a:
        return 0.0
    tot = 0.0
    for s,e in silences:
        lo = max(a, s); hi = min(b, e)
        if hi > lo:
            tot += (hi - lo)
    return max(0.0, tot)

def complement_intervals(silences: List[Tuple[float, float]], total: float) -> List[Tuple[float, float]]:
    """
    [0,total]에서 silences의 여집합(=스피치 구간) 리스트를 반환.
    """
    if total <= 0.0:
        return []
    sil = sorted([(max(0.0, s), min(total, e)) for s, e in silences if e > s and e > 0], key=lambda x: x[0])
    out = []
    cur = 0.0
    for s, e in sil:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        out.append((cur, total))
    return out

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    겹치는 구간 머지. 입력은 (start,end) 리스트. end>start인 것만 사용.
    """
    xs = sorted([(max(0.0, a), max(0.0, b)) for a, b in intervals if b > a], key=lambda x: x[0])
    out: List[Tuple[float, float]] = []
    for s, e in xs:
        if not out or s > out[-1][1] + 1e-6:
            out.append([s, e])  # type: ignore
        else:
            out[-1][1] = max(out[-1][1], e)  # type: ignore
    return [(float(a), float(b)) for a, b in out]

def complement_intervals(intervals: List[Tuple[float, float]], total: float) -> List[Tuple[float, float]]:
    """
    [0,total] 범위에서 intervals의 여집합 반환.
    """
    if total <= 0.0:
        return []
    ints = merge_intervals([(max(0.0, a), min(total, b)) for a, b in intervals if b > a and a < total])
    out: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in ints:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        out.append((cur, total))
    return out