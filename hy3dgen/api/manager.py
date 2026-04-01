import asyncio
import gc
import logging
import threading
import torch
import uuid
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from hy3dgen.inference import ModelWorker
from hy3dgen.api.schemas import JobStatus, JobResponse, JobRequest

logger = logging.getLogger(__name__)

class PriorityRequestManager:
    """
    Manages generation requests with priority queuing and resource cleanup.
    Ensures single-threaded execution of model inference to prevent VRAM OOM.
    """
    def __init__(self, device='cuda', max_concurrent=1):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.jobs: Dict[str, JobResponse] = {}
        self.device = device
        self._shutdown_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        
        # Lazy initialization of the worker to speed up startup
        self.worker: Optional[ModelWorker] = None
        
    async def start(self):
        """Start the background worker loop."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.info("PriorityRequestManager worker started.")

    async def stop(self):
        """Stop the worker loop gracefully."""
        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("PriorityRequestManager worker stopped.")

    async def submit_job(self, request: JobRequest, save_dir: str, priority: int = 10) -> str:
        """
        Submit a job to the queue.
        
        Args:
            request: The generation request (polymorphic)
            save_dir: Directory to save output
            priority: Lower number = higher priority. Default 10.
        
        Returns:
            uid: The unique job ID
        """
        uid = str(uuid.uuid4())
        job = JobResponse(
            uid=uid,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow().isoformat()
        )
        self.jobs[uid] = job
        
        # Queue item: (priority, timestamp, uid, request, save_dir)
        # timestamp acts as secondary sort key for FIFO within same priority
        await self.queue.put((priority, time.time(), uid, request, save_dir))
        logger.info(f"Job {uid} queued with priority {priority}")
        return uid

    def get_job(self, uid: str) -> Optional[JobResponse]:
        return self.jobs.get(uid)

    def cancel_job(self, uid: str):
        if uid in self.jobs:
            # We can only cancel if it's not yet processing (or barely started)
            if self.jobs[uid].status == JobStatus.QUEUED:
                self.jobs[uid].status = JobStatus.CANCELLED
                self.jobs[uid].error = "Cancelled by user"
                logger.info(f"Job {uid} cancelled")

    async def _process_queue(self):
        while not self._shutdown_event.is_set():
            try:
                # Wait for job
                priority, _, uid, request, save_dir = await self.queue.get()
                
                # Check status
                if uid not in self.jobs or self.jobs[uid].status in (JobStatus.CANCELLED, JobStatus.FAILED):
                    self.queue.task_done()
                    continue

                # Run job
                await self._execute_model_worker(uid, request, save_dir)
                
                # Cleanup
                self.queue.task_done()
                self._aggressive_cleanup()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1) # Backoff

    async def _execute_model_worker(self, uid: str, request: JobRequest, save_dir: str):
        job = self.jobs[uid]
        job.status = JobStatus.PROCESSING
        
        try:
            # Initialize worker if needed (Lazy Loading)
            if self.worker is None:
                logger.info("Initializing ModelWorker (Lazy Load)...")
                # Blocking init logic, run in thread to avoid freezing API?
                # Model loading is heavy.
                self.worker = await asyncio.to_thread(
                    ModelWorker, 
                    device=self.device,
                    enable_tex=True, # Archeon default to enabled for now
                    enable_t2i=True
                )
            
            # Prepare params dict from Pydantic model
            params = request.model_dump()
            
            # Map Pydantic fields to ModelWorker expectations
            if request.type == 'text_to_3d':
                params['text'] = request.prompt
            
            # Run generation in thread
            logger.info(f"Starting generation for job {uid}")
            file_path = await asyncio.to_thread(
                self.worker.generate,
                uid,
                params,
                save_dir
            )
            
            job.status = JobStatus.COMPLETED
            job.file_path = file_path
            job.completed_at = datetime.utcnow().isoformat()
            logger.info(f"Job {uid} completed successfully")

        except Exception as e:
            logger.error(f"Job {uid} failed: {e}")
            import traceback
            traceback.print_exc()
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow().isoformat()

    def _aggressive_cleanup(self):
        """Perform aggressive garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
