# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
Hunyuan3D-2GP API Server

A REST API for 3D model generation with:
- MMGP offloading for low-VRAM GPUs
- Job queue with async processing
- Optional API key authentication
- OpenAPI documentation via Pydantic models
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from typing import Optional

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
    MeshSimplifier,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.environ.get('XDG_STATE_HOME', os.path.expanduser('~/.local/state')), 'hy3dgen')
SAVE_DIR = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'hy3dgen', 'api')
handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        filename = os.path.join(LOG_DIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8'
        )
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


# ---------------------------------------------------------------------------
# Pydantic Models (imported from api_models.py)
# ---------------------------------------------------------------------------

from api_models import (
    JobStatus,
    GenerateRequest,
    JobSubmitResponse,
    JobStatusResponse,
    ServerStatusResponse,
    ErrorResponse,
)


# ---------------------------------------------------------------------------
# Job Manager
# ---------------------------------------------------------------------------

class JobManager:
    """Manages async job queue and status tracking."""

    def __init__(self, max_concurrent: int = 2):
        self.jobs: dict[str, dict] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._lock = threading.Lock()
        self._completed_count = 0
        self._semaphore = threading.Semaphore(max_concurrent)

    def create_job(self, params: dict) -> str:
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

    def get_status(self, uid: str) -> Optional[dict]:
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
        """Remove jobs older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []
        for uid, job in self.jobs.items():
            created = datetime.fromisoformat(job['created_at'])
            if created < cutoff and job['status'] in (JobStatus.completed, JobStatus.failed):
                to_remove.append(uid)
                if job.get('file_path') and os.path.exists(job['file_path']):
                    try:
                        os.remove(job['file_path'])
                    except OSError:
                        pass
        for uid in to_remove:
            del self.jobs[uid]


# ---------------------------------------------------------------------------
# Model Worker
# ---------------------------------------------------------------------------

class ModelWorker:
    def __init__(
        self,
        model_path='tencent/Hunyuan3D-2mini',
        tex_model_path='tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        device='cuda',
        enable_tex=False,
        enable_t2i=False,
    ):
        self.model_path = model_path
        self.device = device
        self.enable_tex = enable_tex
        self.enable_t2i = enable_t2i

        logger.info(f"Loading shape generation model {model_path}/{subfolder} ...")
        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        self.pipeline.enable_flashvdm()

        self.pipeline_tex = None
        if enable_tex:
            logger.info(f"Loading texture generation model {tex_model_path} ...")
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)

        self.pipeline_t2i = None
        if enable_t2i:
            from hy3dgen.text2image import HunyuanDiTPipeline
            logger.info("Loading text-to-image model ...")
            self.pipeline_t2i = HunyuanDiTPipeline(
                'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                device=device,
            )

    @torch.inference_mode()
    def generate(self, uid: str, params: dict) -> str:
        # --- Input Processing ---
        if 'image' in params and params['image']:
            image = load_image_from_base64(params['image'])
        elif 'text' in params and params['text']:
            if self.pipeline_t2i is None:
                raise ValueError(
                    "Text-to-3D is not enabled. Start the server with --enable_t23d."
                )
            image = self.pipeline_t2i(params['text'])
        else:
            raise ValueError("No input image or text provided.")

        image = self.rembg(image)

        # --- Mesh Generation ---
        if 'mesh' in params and params['mesh']:
            mesh = trimesh.load(
                BytesIO(base64.b64decode(params['mesh'])), file_type='glb'
            )
        else:
            seed = params.get('seed', 1234)
            generator = torch.Generator(self.device).manual_seed(seed)
            octree_resolution = params.get('octree_resolution', 256)
            num_inference_steps = params.get('num_inference_steps', 5)
            guidance_scale = params.get('guidance_scale', 5.0)

            start_time = time.time()
            mesh = self.pipeline(
                image=image,
                generator=generator,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                mc_algo='dmc',
                output_type='mesh',
            )[0]
            logger.info(f"Shape generation took {time.time() - start_time:.2f}s")

        # --- Post-processing and Texturing ---
        if params.get('texture', False):
            if self.pipeline_tex is None:
                logger.warning("Texture requested but texture model not loaded, skipping.")
            else:
                mesh = FloaterRemover()(mesh)
                mesh = DegenerateFaceRemover()(mesh)
                mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
                start_time = time.time()
                mesh = self.pipeline_tex(mesh, image)
                logger.info(f"Texture generation took {time.time() - start_time:.2f}s")

        # --- Export ---
        file_type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=True) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{uid}.{file_type}')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def cleanup_cache(max_size: int = 200, max_age_hours: int = 24):
    """Remove cache files that exceed count limit or age threshold."""
    if not os.path.exists(SAVE_DIR):
        return

    files = []
    for f in os.listdir(SAVE_DIR):
        filepath = os.path.join(SAVE_DIR, f)
        if os.path.isfile(filepath):
            files.append((filepath, os.path.getmtime(filepath)))

    # Remove files older than max_age_hours
    cutoff_time = time.time() - (max_age_hours * 3600)
    for filepath, mtime in files:
        if mtime < cutoff_time:
            try:
                os.remove(filepath)
                logger.info(f"Cache cleanup: removed old file {filepath}")
            except OSError:
                pass

    # If still over limit, remove oldest
    files = [(f, m) for f, m in files if os.path.exists(f)]
    if len(files) > max_size:
        files.sort(key=lambda x: x[1])
        for filepath, _ in files[:len(files) - max_size]:
            try:
                os.remove(filepath)
                logger.info(f"Cache cleanup: removed excess file {filepath}")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_from_base64(image_b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image_b64)))


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hunyuan3D-2GP API",
    description="REST API for 3D model generation using Hunyuan3D-2, optimized for low-VRAM GPUs.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS restricted to localhost origins for security.
# For remote/Docker usage, override via CORS_ORIGINS env var (comma-separated).
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:*,http://127.0.0.1:*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global references (set in __main__)
worker: Optional[ModelWorker] = None
job_manager: Optional[JobManager] = None
API_KEY: Optional[str] = None


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    if API_KEY is None:
        return  # No auth required
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/generate",
    summary="Generate 3D Model (Synchronous)",
    description="Generate a 3D model from an image or text prompt. Returns the model file directly.",
    responses={
        200: {"description": "Generated 3D model file"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        422: {"description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def generate(request: GenerateRequest, _=Depends(verify_api_key)):
    logger.info("Synchronous generation request received")
    uid = str(uuid.uuid4())
    try:
        file_path = worker.generate(uid, request.model_dump(exclude_none=True))
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.CudaError as e:
        logger.error(f"CUDA error: {e}")
        raise HTTPException(status_code=500, detail="GPU error during generation")
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during generation")


@app.post(
    "/send",
    response_model=JobSubmitResponse,
    summary="Submit Async Generation Job",
    description="Submit a 3D generation job for async processing. Use /status/{uid} to check progress.",
    responses={
        200: {"model": JobSubmitResponse},
        401: {"model": ErrorResponse},
    },
)
async def send_job(request: GenerateRequest, _=Depends(verify_api_key)):
    logger.info("Async generation job submitted")
    params = request.model_dump(exclude_none=True)
    uid = job_manager.create_job(params)

    def _process_job(job_uid, job_params):
        job_manager._semaphore.acquire()
        try:
            with job_manager._lock:
                job_manager._active_count += 1
            job_manager.update_status(job_uid, JobStatus.processing)
            file_path = worker.generate(job_uid, job_params)
            job_manager.update_status(
                job_uid, JobStatus.completed,
                file_path=file_path,
                completed_at=datetime.utcnow().isoformat(),
            )
            job_manager._completed_count += 1
        except Exception as e:
            logger.error(f"Job {job_uid} failed: {e}")
            traceback.print_exc()
            job_manager.update_status(
                job_uid, JobStatus.failed,
                error=str(e),
                completed_at=datetime.utcnow().isoformat(),
            )
        finally:
            with job_manager._lock:
                job_manager._active_count -= 1
            job_manager._semaphore.release()

    threading.Thread(target=_process_job, args=(uid, params), daemon=True).start()
    return JobSubmitResponse(uid=uid, status=JobStatus.queued)


@app.get(
    "/status/{uid}",
    response_model=JobStatusResponse,
    summary="Check Job Status",
    description="Check the status of an async generation job. When completed, includes the model as base64.",
    responses={
        200: {"model": JobStatusResponse},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_job_status(uid: str, _=Depends(verify_api_key)):
    job = job_manager.get_status(uid)

    if job is None:
        # Backward compatibility: check for file in SAVE_DIR
        save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
        if os.path.exists(save_file_path):
            model_b64 = base64.b64encode(open(save_file_path, 'rb').read()).decode()
            return JobStatusResponse(
                uid=uid, status=JobStatus.completed, model_base64=model_b64
            )
        raise HTTPException(status_code=404, detail=f"Job {uid} not found")

    response = JobStatusResponse(
        uid=uid,
        status=job['status'],
        created_at=job.get('created_at'),
        completed_at=job.get('completed_at'),
        error=job.get('error'),
    )

    if job['status'] == JobStatus.completed and job.get('file_path'):
        file_path = job['file_path']
        if os.path.exists(file_path):
            response.model_base64 = base64.b64encode(
                open(file_path, 'rb').read()
            ).decode()

    return response


@app.get(
    "/download/{uid}",
    summary="Download Completed Model",
    description="Download the generated 3D model file directly.",
    responses={
        200: {"description": "3D model file"},
        404: {"model": ErrorResponse},
    },
)
async def download_model(uid: str, _=Depends(verify_api_key)):
    job = job_manager.get_status(uid)
    if job is None or job['status'] != JobStatus.completed or not job.get('file_path'):
        # Fallback: check directly in SAVE_DIR
        for ext in ['glb', 'obj', 'ply', 'stl']:
            path = os.path.join(SAVE_DIR, f'{uid}.{ext}')
            if os.path.exists(path):
                return FileResponse(path)
        raise HTTPException(status_code=404, detail=f"Model for job {uid} not found")

    return FileResponse(job['file_path'])


@app.get(
    "/health",
    response_model=ServerStatusResponse,
    summary="Server Health Check",
    description="Get server status, queue information, and loaded model details.",
)
async def health_check():
    return ServerStatusResponse(
        status="ok",
        queue_length=job_manager.queue_length if job_manager else 0,
        active_jobs=job_manager.active_count if job_manager else 0,
        completed_jobs=job_manager.completed_count if job_manager else 0,
        model_path=worker.model_path if worker else "not loaded",
        has_texture=worker.enable_tex if worker else False,
        has_t2i=worker.enable_t2i if worker else False,
    )


_start_time = time.time()


@app.get(
    "/metrics",
    summary="Runtime Metrics",
    description="Get runtime metrics: uptime, memory usage, GPU info, and job stats.",
)
async def metrics():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
                "memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1),
            }
    except Exception:
        pass

    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "process": {
            "pid": os.getpid(),
            "rss_mb": round(mem_info.rss / 1024**2, 1),
            "vms_mb": round(mem_info.vms / 1024**2, 1),
            "threads": process.num_threads(),
        },
        "gpu": gpu_info,
        "jobs": {
            "queue_length": job_manager.queue_length if job_manager else 0,
            "active": job_manager.active_count if job_manager else 0,
            "completed": job_manager.completed_count if job_manager else 0,
            "max_concurrent": job_manager.max_concurrent if job_manager else 0,
        },
    }


@app.post(
    "/cleanup",
    summary="Trigger Cache Cleanup",
    description="Manually trigger cache cleanup. Removes old files and expired jobs.",
)
async def trigger_cleanup(_=Depends(verify_api_key)):
    cleanup_cache()
    if job_manager:
        job_manager.cleanup_old_jobs()
    return {"status": "cleanup completed"}


# ---------------------------------------------------------------------------
# MMGP Integration Helper
# ---------------------------------------------------------------------------

def setup_mmgp_offloading(worker_instance: ModelWorker, profile: int, verbose: int):
    """Setup MMGP memory offloading for low-VRAM GPUs."""
    try:
        from mmgp import offload

        def replace_property_getter(instance, property_name, new_getter):
            original_class = type(instance)
            original_property = getattr(original_class, property_name)
            custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})
            new_property = property(new_getter, original_property.fset)
            setattr(custom_class, property_name, new_property)
            instance.__class__ = custom_class
            return instance

        replace_property_getter(
            worker_instance.pipeline, "_execution_device", lambda self: "cuda"
        )

        pipe = offload.extract_models("pipeline", worker_instance.pipeline)
        if worker_instance.pipeline_tex:
            pipe.update(offload.extract_models("pipeline_tex", worker_instance.pipeline_tex))
            worker_instance.pipeline_tex.models["multiview_model"].pipeline.vae.use_slicing = True
        if worker_instance.pipeline_t2i:
            pipe.update(offload.extract_models("pipeline_t2i", worker_instance.pipeline_t2i))

        kwargs = {}
        if profile < 5:
            kwargs["pinnedMemory"] = "pipeline/model"
        if profile not in (1, 3):
            kwargs["budgets"] = {"*": 2200}

        offload.default_verboseLevel = verbose
        offload.profile(pipe, profile_no=profile, verboseLevel=verbose, **kwargs)
        logger.info(f"MMGP offloading enabled with profile {profile}")

    except ImportError:
        logger.warning("mmgp not installed, running without memory offloading")
    except Exception as e:
        logger.warning(f"MMGP setup failed: {e}, running without memory offloading")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D-2GP API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (use 0.0.0.0 for Docker/remote)")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini',
                        help="Shape generation model path")
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo',
                        help="Model subfolder")
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2',
                        help="Texture generation model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--limit-model-concurrency", type=int, default=2,
                        help="Max concurrent generation jobs")
    parser.add_argument('--enable_tex', action='store_true',
                        help="Enable texture generation")
    parser.add_argument('--enable_t23d', action='store_true',
                        help="Enable text-to-3D (loads HunyuanDiT)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="API key for authentication (disabled if not set)")
    parser.add_argument('--profile', type=str, default="3",
                        help="MMGP memory offloading profile (1-5)")
    parser.add_argument('--verbose', type=str, default="1",
                        help="Verbose level for MMGP")
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.environ.get('XDG_STATE_HOME', os.path.expanduser('~/.local/state')), 'hy3dgen'),
                        help="Directory for log files (XDG-compliant default)")
    parser.add_argument('--cache_dir', type=str,
                        default=os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'hy3dgen', 'api'),
                        help="Directory for cached/generated files (XDG-compliant default)")
    parser.add_argument('--turbo', action='store_true',
                        help="Use turbo subfolder for faster generation")
    parser.add_argument('--no_mmgp', action='store_true',
                        help="Disable MMGP memory offloading")

    args = parser.parse_args()

    # Configure directories
    LOG_DIR = args.log_dir
    SAVE_DIR = args.cache_dir
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Configure auth
    API_KEY = args.api_key

    # Non-blocking version check
    try:
        from hy3dgen.version import check_for_updates
        update_info = check_for_updates()
        if update_info:
            logger.info(f"Update available: {update_info['latest']} → {update_info['url']}")
    except Exception:
        pass  # Never block startup

    # Setup turbo mode
    subfolder = args.subfolder
    if args.turbo and 'turbo' not in subfolder:
        subfolder = subfolder + '-turbo'

    logger = build_logger("api_server", "api_server.log")
    logger.info(f"Starting Hunyuan3D-2GP API Server with args: {args}")

    # Initialize job manager
    job_manager = JobManager(max_concurrent=args.limit_model_concurrency)

    # Initialize worker
    worker = ModelWorker(
        model_path=args.model_path,
        tex_model_path=args.tex_model_path,
        subfolder=subfolder,
        device=args.device,
        enable_tex=args.enable_tex,
        enable_t2i=args.enable_t23d,
    )

    # Setup MMGP offloading
    if not args.no_mmgp:
        setup_mmgp_offloading(worker, int(args.profile), int(args.verbose))

    # Cleanup old cache on startup
    cleanup_cache()

    logger.info(f"Server ready at http://{args.host}:{args.port}")
    logger.info(f"API docs at http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
