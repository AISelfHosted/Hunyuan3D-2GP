from enum import Enum
from typing import Optional, Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MeshOpsAction(str, Enum):
    DECIMATE = "decimate"
    CONVERT = "convert"


class BaseGenerationRequest(BaseModel):
    """Common parameters for all generation types."""
    seed: int = Field(1234, description="Random seed")
    steps: int = Field(50, ge=1, le=100, description="Denoising steps")
    guidance: float = Field(5.0, ge=1.0, le=20.0, description="Guidance scale")
    octree_resolution: int = Field(256, ge=16, le=512, description="Voxel resolution")
    format: Literal['glb', 'obj', 'ply', 'stl'] = 'glb'
    texture: bool = Field(False, description="Generate texture?")
    face_count: int = Field(40000, ge=100, le=1000000, description="Target face count for reduction")
    
    model_config = ConfigDict(extra='forbid')


class TextTo3DRequest(BaseGenerationRequest):
    type: Literal['text_to_3d'] = 'text_to_3d'
    prompt: str = Field(..., min_length=1, description="Text prompt")


class ImageTo3DRequest(BaseGenerationRequest):
    type: Literal['image_to_3d'] = 'image_to_3d'
    image: str = Field(..., description="Base64 encoded image")
    remove_background: bool = Field(True, description="Remove background using rembg?")


class MultiviewRequest(BaseGenerationRequest):
    type: Literal['multiview'] = 'multiview'
    front: str = Field(..., description="Front view base64")
    back: str = Field(..., description="Back view base64")
    left: str = Field(..., description="Left view base64")
    right: str = Field(..., description="Right view base64")


# Discriminated Union for polymorphic handling
JobRequest = Annotated[Union[TextTo3DRequest, ImageTo3DRequest, MultiviewRequest], Field(discriminator='type')]


class JobResponse(BaseModel):
    uid: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    error: str
    code: int = 400


class MeshOpsRequest(BaseModel):
    job_uid: str
    action: MeshOpsAction
    format: str = 'glb'
    ratio: float = 0.5
    model_config = ConfigDict(use_enum_values=True)
