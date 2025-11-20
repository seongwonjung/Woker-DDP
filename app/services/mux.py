# mux.py
import json
import os
import subprocess
from pydub import AudioSegment
from pathlib import Path

try:
    from app.configs import get_job_paths
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from configs import get_job_paths


def mux_audio_video(job_id: str, video_input_path: Path | None = None):
    """합성 음성과 배경음을 결합하고 원본 영상에 다시 입혀 최종 영상을 생성합니다."""
    paths = get_job_paths(job_id)
    background_path = paths.vid_bgm_dir / "background.wav"

    # Sync 단계의 결과물인 segments_synced.json을 우선적으로 사용
    synced_meta_path = paths.vid_tts_dir / "synced" / "segments_synced.json"
    tts_meta_path = paths.vid_tts_dir / "segments.json"

    if synced_meta_path.is_file():
        meta_path = synced_meta_path
    elif tts_meta_path.is_file():
        meta_path = tts_meta_path
    else:
        raise FileNotFoundError("TTS or Synced metadata file not found.")

    video_input = video_input_path or (paths.input_dir / "source.mp4")
    video_input = Path(video_input)
    if not video_input.is_file():
        raise RuntimeError(f"Original video file not found for muxing at {video_input}")
    if not background_path.is_file():
        raise FileNotFoundError("Background audio not found. Run Demucs stage.")

    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise ValueError("No segments found in metadata file.")

    # 배경 오디오 로드
    background_audio = AudioSegment.from_wav(str(background_path))
    total_duration_ms = len(background_audio)
    # 보컬 합성 결과를 따로 쌓은 뒤 마지막에 배경과 합쳐
    # 배경 트랙 볼륨이 자동으로 낮아지는 일을 방지한다.
    voice_mix = AudioSegment.silent(duration=total_duration_ms)

    # 메타데이터를 기반으로 음성 구간을 정확한 위치에 오버레이
    for segment in segments:
        audio_file_path = Path(segment["audio_file"])
        if not audio_file_path.is_file():
            print(f"Warning: Audio file not found, skipping: {audio_file_path}")
            continue

        segment_audio = AudioSegment.from_wav(str(audio_file_path))

        # 메타데이터에서 정확한 시작 시간 가져오기
        start_time = float(segment.get("start", 0.0))
        start_ms = int(start_time * 1000)

        if start_ms < 0:
            start_ms = 0

        # 해당 위치에 음성 구간을 오버레이 (배경은 나중에 결합)
        voice_mix = voice_mix.overlay(segment_audio, position=start_ms)

    # 배경과 음성 레이어를 마지막에 결합. gain_during_overlay=0 으로
    # 배경이 자동으로 감쇄되지 않도록 명시한다.
    final_audio = background_audio.overlay(voice_mix, position=0, gain_during_overlay=0)

    # 필요 시 패딩/트리밍으로 길이를 배경 오디오와 동일하게 맞춤
    if len(final_audio) < total_duration_ms:
        silence = AudioSegment.silent(duration=(total_duration_ms - len(final_audio)))
        final_audio = final_audio + silence
    elif len(final_audio) > total_duration_ms:
        final_audio = final_audio[:total_duration_ms]

    # 믹싱된 오디오를 outputs 디렉터리에 저장
    output_dir = paths.outputs_vid_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_path = output_dir / "dubbed_audio.wav"
    final_audio.export(str(final_audio_path), format="wav")

    # 원본 영상과 새 오디오를 결합
    output_video_path = output_dir / "dubbed_video.mp4"
    # ffmpeg로 영상의 오디오 트랙을 교체
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_input),
        "-i",
        str(final_audio_path),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_video_path),
    ]
    subprocess.run(cmd, check=True, timeout=600)  # 10분 타임아웃
    return {
        "output_video": str(output_video_path),
        "output_audio": str(final_audio_path),
    }
