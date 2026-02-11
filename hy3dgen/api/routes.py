from fastapi import APIRouter, Depends, HTTPException
import os
import os
import asyncio
from typing import List, Dict
from hy3dgen.api.schemas import JobRequest, JobResponse, MeshOpsRequest
from hy3dgen.api.deps import get_manager, get_mesh_processor
from hy3dgen.api.manager import PriorityRequestManager
from hy3dgen.meshops.processor import MeshProcessor
from hy3dgen.monitoring import get_system_metrics
from hy3dgen.api.config import SAVE_DIR

router = APIRouter(prefix="/v1", tags=["generation"])

# Default save directory (following XDG specs)
# We might want to pass this via config later

@router.post("/jobs", response_model=JobResponse, status_code=202)
async def submit_job(
    request: JobRequest,
    manager: PriorityRequestManager = Depends(get_manager)
):
    """
    Submit a generation job.
    Accepts polymorphic JSON body: { "type": "text_to_3d" | "image_to_3d" | "multiview", ... }
    """
    uid = await manager.submit_job(request, SAVE_DIR)
    
    # Return initial status
    job = manager.get_job(uid)
    return job

@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    manager: PriorityRequestManager = Depends(get_manager)
):
    """List all jobs in memory."""
    return list(manager.jobs.values())

@router.get("/jobs/{uid}", response_model=JobResponse)
async def get_job_status(
    uid: str,
    manager: PriorityRequestManager = Depends(get_manager)
):
    """Retrieve job status and result path."""
    job = manager.get_job(uid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.delete("/jobs/{uid}")
async def cancel_job(
    uid: str,
    manager: PriorityRequestManager = Depends(get_manager)
):
    """Request job cancellation."""
    manager.cancel_job(uid)
    return {"status": "cancellation_requested", "uid": uid}

@router.get("/system/metrics", tags=["system"])
async def get_metrics():
    """Get system resource usage (CPU, GPU, RAM)."""
    return get_system_metrics()

@router.post("/meshops/process")
async def process_mesh(
    request: MeshOpsRequest,
    manager: PriorityRequestManager = Depends(get_manager),
    processor: MeshProcessor = Depends(get_mesh_processor)
):
    """
    Process an existing job's output mesh (Decimate/Convert).
    """
    # 1. Get job
    job = manager.get_job(request.job_uid)
    if not job:
         raise HTTPException(status_code=404, detail="Job not found")
    if not job.file_path or not os.path.exists(job.file_path):
         raise HTTPException(status_code=404, detail="Job result file not found")
    
    # 2. Determine output path
    base_name, _ = os.path.splitext(job.file_path)
    if request.action == 'decimate':
        suffix = f"_decimate_{request.ratio:.2f}"
    else:
        suffix = f"_{request.action}"
    
    output_path = f"{base_name}{suffix}.{request.format}"
    
    # 3. Process (Offload to thread)
    try:
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(
            None, 
            processor.process, 
            job.file_path, 
            output_path, 
            request.action, 
            request.model_dump()
        )
        return {"file_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
