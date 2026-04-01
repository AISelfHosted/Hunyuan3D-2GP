"""
Tests for the API server Pydantic models and validation.
These tests don't require GPU or model loading — they import from api_models.py.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pydantic import ValidationError

from api_models import (
    GenerateRequest,
    JobStatus,
    JobSubmitResponse,
    JobStatusResponse,
    ServerStatusResponse,
    ErrorResponse,
)


def test_generate_request_defaults():
    """Test GenerateRequest with default values."""
    req = GenerateRequest(image="dGVzdA==")
    assert req.seed == 1234
    assert req.octree_resolution == 256
    assert req.num_inference_steps == 5
    assert req.guidance_scale == 5.0
    assert req.texture is False
    assert req.type == "glb"


def test_generate_request_custom_values():
    """Test GenerateRequest with custom values."""
    req = GenerateRequest(
        text="A cute cat",
        seed=42,
        octree_resolution=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        texture=True,
        face_count=10000,
        type="obj",
    )
    assert req.text == "A cute cat"
    assert req.seed == 42
    assert req.octree_resolution == 512
    assert req.texture is True
    assert req.type == "obj"


def test_generate_request_validation():
    """Test that GenerateRequest validates field constraints."""
    # Should fail: octree_resolution out of range
    with pytest.raises(ValidationError):
        GenerateRequest(image="dGVzdA==", octree_resolution=0)

    with pytest.raises(ValidationError):
        GenerateRequest(image="dGVzdA==", octree_resolution=1024)


def test_generate_request_guidance_scale_validation():
    """Test guidance_scale boundaries."""
    with pytest.raises(ValidationError):
        GenerateRequest(image="dGVzdA==", guidance_scale=0.5)

    with pytest.raises(ValidationError):
        GenerateRequest(image="dGVzdA==", guidance_scale=25.0)


def test_job_status_enum():
    """Test JobStatus enum values."""
    assert JobStatus.queued == "queued"
    assert JobStatus.processing == "processing"
    assert JobStatus.completed == "completed"
    assert JobStatus.failed == "failed"


def test_job_submit_response():
    """Test JobSubmitResponse model."""
    resp = JobSubmitResponse(uid="test-uid-123", status=JobStatus.queued)
    assert resp.uid == "test-uid-123"
    assert resp.status == JobStatus.queued

    data = resp.model_dump()
    assert data["uid"] == "test-uid-123"
    assert data["status"] == "queued"


def test_job_status_response():
    """Test JobStatusResponse serialization."""
    resp = JobStatusResponse(
        uid="abc-123",
        status=JobStatus.completed,
        created_at="2025-01-01T00:00:00",
        completed_at="2025-01-01T00:01:00",
        model_base64="dGVzdA==",
    )
    data = resp.model_dump()
    assert data["status"] == "completed"
    assert data["model_base64"] == "dGVzdA=="
    assert data["error"] is None


def test_server_status_response():
    """Test ServerStatusResponse model."""
    resp = ServerStatusResponse(
        status="ok",
        queue_length=3,
        active_jobs=1,
        completed_jobs=10,
        model_path="tencent/Hunyuan3D-2mini",
        has_texture=True,
        has_t2i=False,
    )
    assert resp.queue_length == 3
    assert resp.has_texture is True
    assert resp.has_t2i is False


def test_error_response():
    """Test ErrorResponse model."""
    resp = ErrorResponse(error="Something went wrong", error_code=42)
    assert resp.error == "Something went wrong"
    assert resp.error_code == 42


def test_generate_request_empty_is_valid():
    """Test that a request with no image/text/mesh is valid at model level.
    The API endpoint handles requiring at least one input."""
    req = GenerateRequest()
    assert req.image is None
    assert req.text is None
    assert req.mesh is None
