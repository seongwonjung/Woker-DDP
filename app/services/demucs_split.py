# demucs_split.py
import subprocess
import shutil

from configs import get_job_paths


def split_vocals(job_id: str):
    """Demucs 두 스템 모드로 보컬과 배경음을 분리합니다."""
    paths = get_job_paths(job_id)
    job_dir = paths.interim_dir
    audio_path = paths.vid_speaks_dir / "audio.wav"
    if not audio_path.is_file():
        raise FileNotFoundError("Audio not found. Run ASR stage first.")
    output_dir = job_dir / "demucs_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CUDA를 사용해 Demucs 두 스템(보컬/배경) 분리 실행
    cmd = [
        "python3",
        "-m",
        "demucs.separate",
        "-d",
        "cuda",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "-o",
        str(output_dir),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)

    # Demucs는 모델 이름 폴더 하위에 결과를 생성함
    # 예: interim/<job_id>/demucs_out/htdemucs/audio/vocals.wav
    # 생성된 폴더를 찾아 실제 파일 경로 확정
    demucs_model_dir = output_dir / "htdemucs"
    # 입력 파일명(확장자 제거)과 동일한 폴더가 생성됨
    base_name = audio_path.stem
    sep_dir = demucs_model_dir / base_name

    # 기대되는 출력 파일 경로 지정
    vocals_path = sep_dir / "vocals.wav"
    background_path = sep_dir / "no_vocals.wav"
    if not vocals_path.is_file() or not background_path.is_file():
        raise RuntimeError("Demucs output files not found")
    # interim 디렉터리로 복사해 이후 단계가 쉽게 접근하도록 함
    target_vocals = paths.vid_speaks_dir / "vocals.wav"
    target_background = paths.vid_bgm_dir / "background.wav"
    shutil.copy(vocals_path, target_vocals)
    shutil.copy(background_path, target_background)
    return {
        "vocals": str(target_vocals),
        "background": str(target_background),
    }
