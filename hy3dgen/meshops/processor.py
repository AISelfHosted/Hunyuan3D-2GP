import os
import trimesh
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MeshProcessor:
    def __init__(self):
        pass

    def load_mesh(self, file_path: str) -> trimesh.Trimesh:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mesh file not found: {file_path}")
        
        try:
            # Force 'mesh' to avoid getting a Scene object for single GLBs
            mesh = trimesh.load(file_path, force='mesh')
            return mesh
        except Exception as e:
            logger.error(f"Failed to load mesh {file_path}: {e}")
            raise e

    def decimate(self, mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
        """
        Reduce polygon count.
        ratio: 0.1 to 1.0 (target fraction of faces).
        """
        if ratio >= 1.0 or ratio <= 0.0:
            return mesh
        
        target_faces = int(len(mesh.faces) * ratio)
        logger.info(f"Decimating mesh from {len(mesh.faces)} to {target_faces} faces")
        
        try:
            # simplify_quadratic_decimation is the best balance
            simplified = mesh.simplify_quadratic_decimation(target_faces)
            return simplified
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
            raise e

    def process(self, input_path: str, output_path: str, action: str, params: dict) -> str:
        """
        Load, process, and save mesh.
        """
        mesh = self.load_mesh(input_path)
        
        if action == 'decimate':
            ratio = params.get('ratio', 0.5)
            mesh = self.decimate(mesh, ratio)
        elif action == 'convert':
            pass # Just loading and saving converts format based on extension
            
        # Export
        # Trimesh exports based on file extension
        mesh.export(output_path)
        logger.info(f"Saved processed mesh to {output_path}")
        return output_path
