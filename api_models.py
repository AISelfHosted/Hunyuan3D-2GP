"""
Pydantic models for the Hunyuan3D-2GP API Server.
Provides request/response schemas and OpenAPI documentation.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Status of an async generation job."""
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class GenerateRequest(BaseModel):
    """Request body for 3D model generation."""
    image: Optional[str] = Field(None, description="Base64 encoded input image")
    text: Optional[str] = Field(None, description="Text prompt for text-to-3D generation")
    mesh: Optional[str] = Field(None, description="Base64 encoded GLB mesh for re-texturing")
    seed: int = Field(1234, description="Random seed for reproducibility")
    octree_resolution: int = Field(256, ge=16, le=512, description="Octree resolution for mesh generation")
    num_inference_steps: int = Field(5, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(5.0, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    texture: bool = Field(False, description="Whether to generate texture for the model")
    face_count: int = Field(40000, ge=100, le=1000000, description="Target face count for mesh reduction")
    type: str = Field("glb", description="Output file format (glb, obj, ply, stl)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "<base64_encoded_image>",
                "seed": 1234,
                "octree_resolution": 256,
                "num_inference_steps": 5,
                "guidance_scale": 5.0,
                "texture": False,
                "type": "glb",
            }
        }
    )


class JobSubmitResponse(BaseModel):
    """Response when submitting an async job."""
    uid: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(JobStatus.queued, description="Initial job status")


class JobStatusResponse(BaseModel):
    """Response for job status queries."""
    uid: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: Optional[str] = Field(None, description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    model_base64: Optional[str] = Field(None, description="Base64 encoded model (only when completed)")


class ServerStatusResponse(BaseModel):
    """Server health and status information."""
    status: str = Field("ok", description="Server status")
    queue_length: int = Field(0, description="Number of jobs in queue")
    active_jobs: int = Field(0, description="Number of actively processing jobs")
    completed_jobs: int = Field(0, description="Total completed jobs")
    model_path: str = Field(..., description="Loaded model path")
    has_texture: bool = Field(False, description="Whether texture generation is available")
    has_t2i: bool = Field(False, description="Whether text-to-image is available")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    error_code: int = Field(1, description="Error code")
