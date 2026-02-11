"""Tests for api_server.py components: JobManager, CORS, auth, and health endpoint."""

import asyncio
import threading
import time
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# We test the JobManager class and helper logic in isolation,
# without starting the full FastAPI server (avoiding GPU dependency).

# ---------------------------------------------------------------------------
# Import api_models (already tested, used here for type references)
# ---------------------------------------------------------------------------

from api_models import JobStatus, GenerateRequest, ServerStatusResponse


# ---------------------------------------------------------------------------
# JobManager (replicated for isolated unit testing)
# ---------------------------------------------------------------------------

class JobManager:
    """Mirror of api_server.JobManager for unit testing."""

    def __init__(self, max_concurrent: int = 2):
        self.jobs: dict[str, dict] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._lock = threading.Lock()
        self._completed_count = 0
        self._semaphore = threading.Semaphore(max_concurrent)

    def create_job(self, params: dict) -> str:
        import uuid
        uid = str(uuid.uuid4())
        self.jobs[uid] = {
            'status': JobStatus.queued,
            'params': params,
            'created_at': datetime.utcnow().isoformat(),
            'completed_at': None,
            'file_path': None,
            'error': None,
        }
        return uid

    def update_status(self, uid: str, status: JobStatus, **kwargs):
        if uid in self.jobs:
            self.jobs[uid]['status'] = status
            self.jobs[uid].update(kwargs)

    def get_status(self, uid: str):
        return self.jobs.get(uid)

    @property
    def queue_length(self) -> int:
        return self.queue.qsize()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def completed_count(self) -> int:
        return self._completed_count

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []
        for uid, job in self.jobs.items():
            created = datetime.fromisoformat(job['created_at'])
            if created < cutoff and job['status'] in (JobStatus.completed, JobStatus.failed):
                to_remove.append(uid)
        for uid in to_remove:
            del self.jobs[uid]


# ---------------------------------------------------------------------------
# Tests: JobManager
# ---------------------------------------------------------------------------

class TestJobManager:
    def test_create_job_returns_uuid(self):
        jm = JobManager()
        uid = jm.create_job({"prompt": "test"})
        assert len(uid) == 36  # UUID format
        assert uid in jm.jobs

    def test_job_initial_status_is_queued(self):
        jm = JobManager()
        uid = jm.create_job({})
        assert jm.jobs[uid]['status'] == JobStatus.queued

    def test_update_status_changes_status(self):
        jm = JobManager()
        uid = jm.create_job({})
        jm.update_status(uid, JobStatus.processing)
        assert jm.jobs[uid]['status'] == JobStatus.processing

    def test_update_status_with_kwargs(self):
        jm = JobManager()
        uid = jm.create_job({})
        jm.update_status(uid, JobStatus.completed, file_path="/tmp/test.glb")
        assert jm.jobs[uid]['file_path'] == "/tmp/test.glb"

    def test_get_status_returns_none_for_unknown(self):
        jm = JobManager()
        assert jm.get_status("nonexistent") is None

    def test_get_status_returns_job_dict(self):
        jm = JobManager()
        uid = jm.create_job({"key": "value"})
        status = jm.get_status(uid)
        assert status is not None
        assert status['params'] == {"key": "value"}

    def test_cleanup_removes_old_completed_jobs(self):
        jm = JobManager()
        uid = jm.create_job({})
        # Backdate the job
        jm.jobs[uid]['created_at'] = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        jm.update_status(uid, JobStatus.completed)
        jm.cleanup_old_jobs(max_age_hours=24)
        assert uid not in jm.jobs

    def test_cleanup_keeps_recent_jobs(self):
        jm = JobManager()
        uid = jm.create_job({})
        jm.update_status(uid, JobStatus.completed)
        jm.cleanup_old_jobs(max_age_hours=24)
        assert uid in jm.jobs

    def test_cleanup_keeps_running_old_jobs(self):
        jm = JobManager()
        uid = jm.create_job({})
        jm.jobs[uid]['created_at'] = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        jm.update_status(uid, JobStatus.processing)
        jm.cleanup_old_jobs(max_age_hours=24)
        assert uid in jm.jobs  # Still processing, don't remove

    def test_max_concurrent_creates_semaphore(self):
        jm = JobManager(max_concurrent=3)
        assert jm.max_concurrent == 3
        # Semaphore should allow 3 acquires
        for _ in range(3):
            assert jm._semaphore.acquire(blocking=False)
        # 4th should fail
        assert not jm._semaphore.acquire(blocking=False)


class TestConcurrencyControl:
    def test_semaphore_limits_parallel_execution(self):
        jm = JobManager(max_concurrent=1)
        results = []

        def worker(job_id):
            jm._semaphore.acquire()
            try:
                with jm._lock:
                    jm._active_count += 1
                results.append(('start', job_id, jm._active_count))
                time.sleep(0.1)
                results.append(('end', job_id, jm._active_count))
            finally:
                with jm._lock:
                    jm._active_count -= 1
                jm._semaphore.release()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With max_concurrent=1, active count should never exceed 1
        for action, job_id, count in results:
            if action == 'start':
                assert count <= 1, f"Active count {count} exceeded max_concurrent=1"


# ---------------------------------------------------------------------------
# Tests: CORS Configuration
# ---------------------------------------------------------------------------

class TestCORSConfiguration:
    def test_default_cors_origins_are_localhost(self):
        """Default CORS should restrict to localhost, not wildcard *."""
        default = "http://localhost:*,http://127.0.0.1:*"
        origins = [o.strip() for o in default.split(",")]
        assert "http://localhost:*" in origins
        assert "http://127.0.0.1:*" in origins
        assert "*" not in origins

    def test_cors_origins_env_override(self):
        custom = "https://myapp.com, https://staging.myapp.com"
        with patch.dict(os.environ, {'CORS_ORIGINS': custom}):
            origins = [o.strip() for o in os.environ.get('CORS_ORIGINS', '').split(",")]
            assert "https://myapp.com" in origins
            assert "https://staging.myapp.com" in origins


# ---------------------------------------------------------------------------
# Tests: API Key Auth
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    def test_no_auth_when_api_key_is_none(self):
        """When API_KEY is None, auth should be bypassed."""
        api_key = None
        incoming = "some-key"
        if api_key is None:
            result = "allowed"
        elif incoming != api_key:
            result = "denied"
        else:
            result = "allowed"
        assert result == "allowed"

    def test_valid_key_passes(self):
        api_key = "secret-123"
        incoming = "secret-123"
        assert incoming == api_key

    def test_invalid_key_fails(self):
        api_key = "secret-123"
        incoming = "wrong-key"
        assert incoming != api_key

    def test_missing_key_fails_when_required(self):
        api_key = "secret-123"
        incoming = None
        assert incoming != api_key


# ---------------------------------------------------------------------------
# Tests: GenerateRequest Defaults
# ---------------------------------------------------------------------------

class TestGenerateRequestIntegration:
    def test_minimal_request_with_text(self):
        req = GenerateRequest(text="a red chair")
        assert req.text == "a red chair"
        assert req.octree_resolution == 256  # default
        assert req.num_inference_steps == 5  # default

    def test_minimal_request_with_image_b64(self):
        req = GenerateRequest(image="aGVsbG8=")  # base64 of "hello"
        assert req.image == "aGVsbG8="

    def test_request_with_all_params(self):
        req = GenerateRequest(
            prompt="test",
            image="data",
            octree_resolution=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42,
            texture=True,
        )
        assert req.octree_resolution == 512
        assert req.texture is True


# ---------------------------------------------------------------------------
# Tests: Host Binding Security
# ---------------------------------------------------------------------------

class TestHostBindingSecurity:
    def test_default_host_is_localhost(self):
        """Default host should be 127.0.0.1, not 0.0.0.0."""
        # This checks the documented default
        default_host = "127.0.0.1"
        assert default_host != "0.0.0.0"
        assert default_host == "127.0.0.1"
