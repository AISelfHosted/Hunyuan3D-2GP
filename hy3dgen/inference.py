import os
import time
import base64
import torch
import trimesh
import tempfile
import logging
from io import BytesIO
from PIL import Image

try:
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
        FaceReducer,
        MeshSimplifier,
    )
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
except ImportError:
    # Handle partial imports during install/setup if needed
    pass

logger = logging.getLogger(__name__)

def load_image_from_base64(image_b64: str) -> Image.Image:
    """Helper to load PIL image from base64 string."""
    return Image.open(BytesIO(base64.b64decode(image_b64)))

class ModelWorker:
    """
    Unified worker for 3D generation tasks.
    Handles loading models, running inference pipelines, and post-processing.
    """
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
    def generate(self, uid: str, params: dict, save_dir: str) -> str:
        """
        Run generation pipeline.
        
        Args:
            uid: Unique job ID
            params: Dictionary of parameters (image/text, steps, etc.)
            save_dir: Directory to save output files
            
        Returns:
            Path to the saved model file
        """
        # --- Input Processing ---
        if 'image' in params and params['image']:
            # Handle potential base64 prefix
            img_data = params['image']
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            image = load_image_from_base64(img_data)
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
            mesh_data = params['mesh']
            if ',' in mesh_data:
                mesh_data = mesh_data.split(',')[1]
            mesh = trimesh.load(
                BytesIO(base64.b64decode(mesh_data)), file_type='glb'
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
        
        # Ensure save dir exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{uid}.{file_type}')
        
        # Using tempfile first for safety
        with tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=True) as temp_file:
            mesh.export(temp_file.name)
            # Reload to flush and ensure validity
            mesh = trimesh.load(temp_file.name)
            mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path
