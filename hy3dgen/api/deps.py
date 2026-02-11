from fastapi import Request
from hy3dgen.api.manager import PriorityRequestManager
from hy3dgen.meshops.processor import MeshProcessor

async def get_manager(request: Request) -> PriorityRequestManager:
    """Dependency to retrieve the PriorityRequestManager instance."""
    return request.app.state.manager

def get_mesh_processor(request: Request) -> MeshProcessor:
    return request.app.state.mesh_processor
