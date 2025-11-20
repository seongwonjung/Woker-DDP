import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../app"))

# Mock environment variables
os.environ["AWS_S3_BUCKET"] = "test-bucket"
os.environ["SQS_QUEUE_URL"] = "test-queue-url"

from worker import handle_split_job, handle_chunk_job, handle_merge_job


class TestDistributedFlow(unittest.TestCase):

    @patch("worker.s3_client")
    @patch("worker.sqs_client")
    @patch("worker.download_from_s3")
    @patch("worker.upload_to_s3")
    @patch("worker.subprocess.run")
    @patch("worker.run_diarization")
    @patch("worker.split_audio_by_vad")
    @patch("worker.run_asr")
    def test_full_flow(
        self,
        mock_run_asr,
        mock_split_vad,
        mock_run_diarization,
        mock_subprocess,
        mock_upload,
        mock_download,
        mock_sqs,
        mock_s3,
    ):

        # Mock 설정
        mock_download.return_value = True
        mock_upload.return_value = True

        # VAD 분할 결과 Mock
        mock_split_vad.return_value = [
            {"path": Path("/tmp/chunk_0.wav"), "start": 0.0, "end": 10.0},
            {"path": Path("/tmp/chunk_1.wav"), "start": 10.0, "end": 20.0},
        ]

        # 화자 분리(Diarization) 결과 Mock
        mock_run_diarization.return_value = [
            {"start": 0.5, "end": 5.0, "speaker": "SPEAKER_00"}
        ]

        # STT 결과 Mock
        mock_run_asr.return_value = {"language": "en"}  # 미리보기 결과

        # 조정을 위한 S3 ListObjects Mock
        # 첫 번째 호출 (청크 0): 1개 키 반환
        # 두 번째 호출 (청크 1): 2개 키 반환 (모두 완료됨)
        mock_s3.list_objects_v2.side_effect = [{"KeyCount": 1}, {"KeyCount": 2}]

        # 1. Splitter 트리거
        split_job = {
            "job_type": "FULL_VIDEO_JOB",
            "job_id": "test-job",
            "input_key": "videos/test.mp4",
            "callback_url": "http://callback",
            "source_lang": "en",
        }

        handle_split_job(split_job)

        # Splitter 동작 검증
        mock_split_vad.assert_called_once()
        mock_run_diarization.assert_called_once()
        # 2개의 CHUNK_JOB 메시지를 보내야 함
        self.assertEqual(mock_sqs.send_message.call_count, 2)

        # CHUNK_JOB 페이로드 추출
        chunk_jobs = []
        for call in mock_sqs.send_message.call_args_list:
            body = json.loads(call.kwargs["MessageBody"])
            chunk_jobs.append(body)

        self.assertEqual(len(chunk_jobs), 2)
        self.assertEqual(chunk_jobs[0]["job_type"], "CHUNK_JOB")
        self.assertEqual(chunk_jobs[0]["chunk_index"], 0)

        # 2. Chunk Worker 트리거
        # 청크 0
        handle_chunk_job(chunk_jobs[0])
        mock_run_asr.assert_called()

        # 청크 1
        handle_chunk_job(chunk_jobs[1])

        # 조정 검증 (MERGE_JOB 큐 등록됨)
        # SQS send_message 호출 횟수가 1 증가해야 함 (MERGE_JOB)
        # 총 호출: 2 (Splitter) + 1 (Chunk 1 -> Merge) = 3
        # 참고: 청크 0은 mock_s3가 KeyCount=1을 반환했으므로 병합을 큐에 넣지 않았음
        self.assertEqual(mock_sqs.send_message.call_count, 3)

        merge_job_call = mock_sqs.send_message.call_args_list[-1]
        merge_job = json.loads(merge_job_call.kwargs["MessageBody"])

        self.assertEqual(merge_job["job_type"], "MERGE_JOB")
        self.assertEqual(merge_job["job_id"], "test-job")

        # 3. Merger 트리거
        # 매니페스트와 결과를 위한 파일 읽기 Mock 필요
        with patch(
            "builtins.open", unittest.mock.mock_open(read_data="{}")
        ) as mock_file:
            # 유효한 데이터를 반환하도록 json.load Mock
            with patch("json.load") as mock_json_load:
                mock_json_load.side_effect = [
                    # 매니페스트
                    {
                        "chunks": [
                            {"index": 0, "start": 0.0},
                            {"index": 1, "start": 10.0},
                        ]
                    },
                    # 결과 0
                    [{"start": 0.0, "end": 5.0, "text": "Hello"}],
                    # 결과 1
                    [{"start": 0.0, "end": 5.0, "text": "World"}],
                ]

                handle_merge_job(merge_job)

        # 최종 업로드 검증
        # transcript.json을 업로드해야 함
        # mock_upload 호출:
        # Splitter: Diarization, Chunk 0, Chunk 1, Manifest (4회)
        # Chunk 0: Result (1회)
        # Chunk 1: Result (1회)
        # Merger: Final Transcript (1회)
        # 총 7회
        self.assertTrue(mock_upload.call_count >= 7)


if __name__ == "__main__":
    unittest.main()
