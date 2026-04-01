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

import os
import sys
import random
import shutil
import time
from glob import glob
from pathlib import Path
import webbrowser
from threading import Timer

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from mmgp import offload
import uuid
from hy3dgen.monitoring import get_system_metrics

from hy3dgen.shapegen.utils import logger as _shapegen_logger
from hy3dgen.shapegen.pipelines import export_to_trimesh

import logging
import logging.handlers

# --- Unified logging for Archeon Launcher ---
_XDG_CACHE = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
_XDG_STATE = os.environ.get('XDG_STATE_HOME', os.path.expanduser('~/.local/state'))
_LOG_DIR = os.path.join(_XDG_STATE, 'hy3dgen')
os.makedirs(_LOG_DIR, exist_ok=True)

# Configure global logging
_formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Root logger for all modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.TimedRotatingFileHandler(
            os.path.join(_LOG_DIR, 'launcher.log'), when='D', utc=True, encoding='UTF-8',
        )
    ]
)

logger = logging.getLogger('hy3dgen.launcher')
# Silence some noisy third-party loggers if needed
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

MAX_SEED = int(1e7)


# --- Global Worker Variables (Lazy Loading) ---
rmbg_worker = None
i23d_worker = None
texgen_worker = None
t2i_worker = None
floater_remove_worker = None
degenerate_face_remove_worker = None
face_reduce_worker = None
HAS_TEXTUREGEN = False
HAS_T2I = False


# --- Helper for GPU Poor (mmgp) ---
def replace_property_getter(obj, name, getter):
    type(obj).name = property(fget=getter)


def get_rmbg_worker():
    global rmbg_worker
    if rmbg_worker is None:
        from hy3dgen.rembg import BackgroundRemover
        logger.info("Initializing Background Remover...")
        rmbg_worker = BackgroundRemover()
    return rmbg_worker


def get_shape_worker():
    global i23d_worker
    if i23d_worker is None:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        logger.info(f"Initializing Shape Generator ({args.model_path})...")
        i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            use_safetensors=True,
            device=args.device,
        )
        if args.enable_flashvdm:
            mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
            i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        if args.compile:
            i23d_worker.compile()
        
        # Memory Management for GPU Poor
        replace_property_getter(i23d_worker, "_execution_device", lambda self: "cuda")
        pipe = offload.extract_models("i23d_worker", i23d_worker)
        
        profile = int(args.profile)
        kwargs_offload = {"pinnedMemory": "i23d_worker/model"} if profile < 5 else {}
        if profile != 1 and profile != 3:
            kwargs_offload["budgets"] = {"*": 2200}
        
        offload.default_verboseLevel = int(args.verbose)
        offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs_offload)
    return i23d_worker


def get_texgen_worker():
    global texgen_worker, HAS_TEXTUREGEN
    if texgen_worker is None:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            logger.info(f"Initializing Texture Generator ({args.texgen_model_path})...")
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            HAS_TEXTUREGEN = True
            
            # Memory Management
            pipe = offload.extract_models("texgen_worker", texgen_worker)
            texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
            
            profile = int(args.profile)
            kwargs_offload = {}
            if profile != 1 and profile != 3:
                kwargs_offload["budgets"] = {"*": 2200}
            offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs_offload)
        except Exception as e:
            logger.error(f"Failed to load texture generator: {e}")
            HAS_TEXTUREGEN = False
    return texgen_worker


def get_t2i_worker():
    global t2i_worker, HAS_T2I
    if t2i_worker is None and args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline
        logger.info("Initializing Text-to-Image Generator...")
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True
        # Memory Management
        pipe = offload.extract_models("t2i_worker", t2i_worker)
        offload.profile(pipe, profile_no=int(args.profile), verboseLevel=int(args.verbose))
    return t2i_worker


def get_postprocessors():
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker
    if floater_remove_worker is None:
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        floater_remove_worker = FloaterRemover()
        degenerate_face_remove_worker = DegenerateFaceRemover()
        face_reduce_worker = FaceReducer()
    return floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker


def get_t2i_worker():
    global t2i_worker, HAS_T2I
    if t2i_worker is None: # T2I always available if enabled
        from hy3dgen.text2image import HunyuanDiTPipeline
        logger.info("Initializing Text-to-Image Generator...")
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True
        pipe = offload.extract_models("t2i_worker", t2i_worker)
        offload.profile(pipe, profile_no=int(args.profile), verboseLevel=int(args.verbose))
    return t2i_worker


def get_postprocessors():
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker
    if floater_remove_worker is None:
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        floater_remove_worker = FloaterRemover()
        degenerate_face_remove_worker = DegenerateFaceRemover()
        face_reduce_worker = FaceReducer()
    return floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker


def get_example_img_list():
    logger.info('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    logger.info('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def get_example_mv_list():
    logger.info('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
    for mv_dir in os.listdir(root):
        view_list = []
        for view in ['front', 'back', 'left', 'right']:
            path = os.path.join(root, mv_dir, f'{view}.png')
            if os.path.exists(path):
                view_list.append(path)
            else:
                view_list.append(None)
        mv_list.append(view_list)
    return mv_list


def gen_save_folder(max_size=200):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Get all subdirectory paths
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]

    # If directory count exceeds max_size, delete the oldest one
    if len(dirs) >= max_size:
        # Sort by creation time, oldest first
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        logger.info(f"Removed the oldest folder: {oldest_dir}")

    # Generate a new UUID folder name
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    logger.info(f"Created new folder: {new_folder}")

    return new_folder


def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    # Shape generation options
    mc_level=0.0,
    bounds=1.01,
    eta=0.0,
    min_resolution=63,
    s_churn=0.0,
    s_noise=1.0,
    s_tmin=0.0,
    s_tmax=100.0,
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Please provide at least one view image.")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right

    seed = int(randomize_seed_fn(int(seed), randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}

    if image is None:
        start_time = time.time()
        worker = get_t2i_worker()
        if worker is None:
            raise gr.Error("Text to 3D is disabled.")
        image = worker(caption)
        time_meta['text2image'] = time.time() - start_time

    # Auto-detect MV mode based on inputs
    is_mv_run = (mv_image_front is not None or mv_image_back is not None or 
                 mv_image_left is not None or mv_image_right is not None)

    start_time = time.time()
    rmbg = get_rmbg_worker()
    if is_mv_run:
        # Prepare MV input dict
        image = {}
        if mv_image_front: image['front'] = mv_image_front
        if mv_image_back: image['back'] = mv_image_back
        if mv_image_left: image['left'] = mv_image_left
        if mv_image_right: image['right'] = mv_image_right
        
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                image[k] = rmbg(v.convert('RGB'))
    else:
        if check_box_rembg or image.mode == "RGB":
            image = rmbg(image.convert('RGB'))
    time_meta['remove background'] = time.time() - start_time

    start_time = time.time()
    generator = torch.Generator().manual_seed(int(seed))
    
    # Force correct model path for the current run type if it changed
    # In a real dynamic scenario, we'd swap model_path here.
    # For now, we rely on args but inform the user.
    if is_mv_run and 'mv' not in args.model_path:
        logger.warning("MV inputs detected but model is not in MV mode. Result might be poor.")
    
    outputs = get_shape_worker()(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        mc_level=mc_level,
        box_v=bounds,
        eta=eta,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    return mesh, main_image, save_folder, stats, seed


def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    # Shape generation options
    mc_level=0.0,
    bounds=1.01,
    eta=0.0,
    min_resolution=63,
    s_churn=0.0,
    s_noise=1.0,
    s_tmin=0.0,
    s_tmax=100.0,
    # Texture options
    texture_size=2048,
    render_size=2048,
    bake_exp=4,
    bake_angle_thres=75,
    multiview_steps=30,
    delight_steps=50,
    delight_cfg_image=1.5,
    delight_cfg_text=1.0,
    use_antialias=True,
    # Mesh options
    floater_ratio=0.005,
    quality_thr=1.0,
    preserve_boundary=True,
    boundary_weight=3,
    preserve_normal=True,
    preserve_topology=True,
    auto_clean=True,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
        mc_level=mc_level,
        bounds=bounds,
        eta=eta,
        min_resolution=min_resolution,
        s_churn=s_churn,
        s_noise=s_noise,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
    )
    path = export_mesh(mesh, save_folder, textured=False)

    tmp_time = time.time()
    floater_remove, degenerate_remove, face_reduce = get_postprocessors()
    mesh = face_reduce(mesh)
    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

    # Update texture generation config
    if HAS_TEXTUREGEN:
        texgen_worker.config.texture_size = int(texture_size)
        texgen_worker.config.render_size = int(render_size)
        texgen_worker.config.bake_exp = int(bake_exp)
        texgen_worker.config.bake_angle_thres = int(bake_angle_thres)
        # Note: multiview_steps, delight_steps, delight_cfg_image, delight_cfg_text
        # currently use hardcoded defaults in the utility files

    tmp_time = time.time()
    worker = get_texgen_worker()
    if worker:
        textured_mesh = worker(mesh, image)
    else:
        textured_mesh = mesh
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0

    textured_mesh.metadata['extras'] = stats
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                         textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        gr.update(value=path_textured),
        model_viewer_html_textured,
        stats,
        seed,
    )


def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    # Shape generation options
    mc_level=0.0,
    bounds=1.01,
    eta=0.0,
    min_resolution=63,
    s_churn=0.0,
    s_noise=1.0,
    s_tmin=0.0,
    s_tmax=100.0,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
        mc_level=mc_level,
        bounds=bounds,
        eta=eta,
        min_resolution=min_resolution,
        s_churn=s_churn,
        s_noise=s_noise,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
    )
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def build_app():
    global args, SAVE_DIR, CURRENT_DIR, MV_MODE, TURBO_MODE, HTML_HEIGHT, HTML_WIDTH, \
        HTML_OUTPUT_PLACEHOLDER, INPUT_MESH_HTML, example_is, example_ts, example_mvs, SUPPORTED_FORMATS, \
        HAS_TEXTUREGEN, HAS_T2I

    archeon_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0b0f19",
        body_background_fill_dark="#0b0f19",
        block_background_fill="#111827",
        block_background_fill_dark="#111827",
        block_border_width="1px",
        block_title_text_color="#94a3b8",
        button_primary_background_fill="#6366f1",
        button_primary_background_fill_hover="#4f46e5",
        button_primary_text_color="white",
        input_background_fill="#1f2937",
        input_border_color="#374151",
        input_border_color_focus="#6366f1",
    )

    custom_css = """
    .app.svelte-wpkpf6 {
        max-width: 98% !important;
        background-color: #0b0f19 !important;
        margin: 0 auto;
    }
    
    .gradio-container {
        font-family: 'Outfit', sans-serif !important;
    }

    .tabs > .tab-nav > button {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }

    /* Standardized Borders & Cohesion */
    .gr-panel, .gr-block, .gr-box, .gr-form {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Premium Button Effects */
    .gr-button-primary {
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
        filter: brightness(1.1);
    }

    /* Tab Bar Refinement */
    .tabs > .tab-nav {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 15px !important;
    }

    .tabs > .tab-nav > button {
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
        color: #94a3b8 !important;
    }

    .tabs > .tab-nav > button.selected {
        color: white !important;
        border-bottom: 2px solid #6366f1 !important;
        background: rgba(99, 102, 241, 0.05) !important;
    }

    /* Column Alignment & Fixed Height Panels */
    #gen_mesh_panel, #export_mesh_panel {
        height: 640px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        overflow: hidden !important;
    }
    
    /* Target only the middle and right column for min-height to allow left column to stay compact */
    #tabs_output + div, #gallery + div {
        min-height: 640px !important;
    }
    
    /* Limit Gallery Height to ~6 rows (matches 3D viewer) */
    #gallery .gallery {
        max-height: 540px !important;
        overflow-y: auto !important;
    }
    
    #tab_img_gallery, #tab_txt_gallery, #tab_mv_gallery {
        padding-bottom: 20px;
    }
    
    /* Force Gallery to hide pager and fill space */
    .gr-samples-pager {
        display: none !important;
    }

    .gr-samples {
        height: 100% !important;
    }

    /* Clean left column spacing */
    #left-column {
        gap: 8px !important;
        display: flex !important;
        flex-direction: column !important;
    }

    #left-column > div {
        margin-bottom: 0px !important;
    }

    #left-column .gr-group {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    """

    with gr.Blocks(theme=archeon_theme, title='Archeon 3D Launcher', analytics_enabled=False, css=custom_css) as demo:
        with gr.Column(elem_id="header-container"):
            gr.HTML(f"""
            <div style="text-align: center; margin-bottom: 0.2rem; margin-top: 0.2rem;">
                <h1 style="font-size: 1.8rem; margin-bottom: 0px;" class="archeon-header">ARCHEON 3D</h1>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=4, elem_id="left-column"):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=300, sources=['upload', 'clipboard'])

                    with gr.Tab('Text Prompt', id='tab_txt_prompt') as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', id='tab_mv_prompt') as tab_mv:
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image', sources=['upload', 'clipboard'])
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image', sources=['upload', 'clipboard'])
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image', sources=['upload', 'clipboard'])
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image', sources=['upload', 'clipboard'])

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(label='Generation Mode',
                                            info='Turbo = fastest (5 steps), good for most models. Fast = medium speed, better for complex shapes with fine details. Standard = slowest but highest quality, rarely needed.',
                                            choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                        decode_mode = gr.Radio(label='Output Detail Level',
                                               info='How detailed the final 3D mesh will be. Low = fewer polygons, faster. Standard = balanced (recommended). High = most detail but larger file size.',
                                               choices=['Low', 'Standard', 'High'],
                                               value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        gr.Markdown("**Basic Settings** - Start here for the most important options")
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100,
                                                         info='Automatically removes the background from your input image. Turn OFF if your image already has a transparent background.')
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100,
                                                        info='Creates a unique 3D model each time. Turn OFF to reproduce the exact same result.')
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                            info='A number that controls randomness. Same seed + same settings = same 3D model. Useful for reproducing good results.'
                        )
                        gr.Markdown("**Quality vs Speed** - Higher values = better quality but slower")
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps',
                                                  info='How many times the AI refines the 3D model. More steps = better quality but slower. Turbo mode: 5 is good. Normal mode: 20-50.')
                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution',
                                                         info='Controls the detail level of the 3D mesh. Higher = more polygons and finer details. 256 is balanced, 384-512 for maximum detail.')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100,
                                                 info='How closely the AI follows your input image. Higher = more faithful to input but may look stiff. Lower = more creative but may drift from input. 5-7 is usually best.')
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100,
                                                   info='Memory optimization setting. Higher values use more VRAM but process faster. Lower if you get out-of-memory errors.')
                    with gr.Tab('Shape Options', id='tab_shape_options'):
                        gr.Markdown("**Surface Extraction** - Controls how the 3D surface is created from the AI's output")
                        with gr.Row():
                            mc_level = gr.Slider(minimum=-0.1, maximum=0.1, value=0.0, step=0.01,
                                                label='MC Level',
                                                info='Adjusts the "thickness" of surfaces. Negative values make the model slightly larger/puffier. Positive values make it tighter/thinner. Try small changes like 0.02.')
                            bounds = gr.Slider(minimum=0.9, maximum=1.5, value=1.01, step=0.01,
                                              label='Bounds',
                                              info='Size of the 3D workspace. Larger values give the model more room to spread out. Smaller values pack more detail into a tighter space. Usually leave at 1.01.')
                        gr.Markdown("**Randomness Control** - Add variety to your generations (advanced)")
                        with gr.Row():
                            eta = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                                           label='Eta (Randomness)',
                                           info='At 0, the same seed always gives the exact same result. Increase to add creative variation - but results become less predictable. Start with 0.')
                            min_resolution = gr.Slider(minimum=31, maximum=127, value=63, step=2,
                                                      label='Min Resolution',
                                                      info='Starting detail level for building the 3D model. Higher values capture finer details but take longer. 63 is a good balance.')
                        with gr.Row():
                            s_churn = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                                               label='S Churn',
                                               info='Adds "creative noise" during generation. Can help with variety but too high may create artifacts. Keep at 0 for consistent results.')
                            s_noise = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                                               label='S Noise',
                                               info='Strength of the creative noise (only matters if S Churn > 0). Higher = more variation. 1.0 is default.')
                        with gr.Row():
                            s_tmin = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                                              label='S Tmin',
                                              info='When to START adding noise during generation. Higher = noise only affects early stages. Usually leave at 0.')
                            s_tmax = gr.Slider(minimum=1.0, maximum=100.0, value=100.0, step=1.0,
                                              label='S Tmax',
                                              info='When to STOP adding noise during generation. Lower = noise only affects later stages. Usually leave at 100.')
                        gr.Markdown("**Turbo Mode Options** - Only used when running with --turbo flag")
                        with gr.Row():
                            adaptive_kv = gr.Checkbox(value=True, label='Adaptive KV Selection',
                                                     info='Smart memory optimization that adjusts to your model complexity. Leave ON for best results. Only turn OFF if you experience issues.')
                            topk_mode = gr.Dropdown(choices=['mean', 'merge'], value='mean', label='TopK Mode',
                                                   info='"mean" gives smoother, cleaner surfaces. "merge" can be sharper but may have more noise. Try "mean" first.')
                            mini_grid_num = gr.Slider(minimum=2, maximum=8, value=4, step=1,
                                                     label='Mini Grid Num',
                                                     info='How many pieces to split processing into. Higher = faster but uses more VRAM. Lower if you get memory errors.')
                    with gr.Tab('Texture Options', id='tab_texture_options', visible=HAS_TEXTUREGEN):
                        gr.Markdown("**Resolution Settings** - Higher = sharper textures but slower and more VRAM")
                        with gr.Row():
                            texture_size = gr.Dropdown(choices=[512, 1024, 2048, 4096], value=2048,
                                                      label='Texture Size',
                                                      info='Resolution of the final texture image in pixels. 1024=fast/low quality, 2048=balanced (recommended), 4096=best quality but slow and needs lots of VRAM.')
                            render_size = gr.Dropdown(choices=[512, 1024, 2048], value=2048,
                                                     label='Render Size',
                                                     info='Resolution used when generating texture views. Should match or be close to Texture Size. Lower values speed up generation.')
                        gr.Markdown("**Texture Blending** - Controls how colors from different viewing angles are combined")
                        with gr.Row():
                            bake_exp = gr.Slider(minimum=1, maximum=8, value=4, step=1,
                                                label='Bake Sharpness',
                                                info='How sharply to blend texture views. Higher = sharper details but may show visible seams between views. Lower = smoother blending but softer details. 4 is balanced.')
                            bake_angle_thres = gr.Slider(minimum=30, maximum=85, value=75, step=5,
                                                        label='View Angle Limit',
                                                        info='Maximum angle (in degrees) for using a texture view. Lower = only uses straight-on views (sharper but may miss areas). Higher = uses angled views too (better coverage). 70-80 is good.')
                        gr.Markdown("**View Generation** - Creates images of your model from multiple angles to build the texture")
                        with gr.Row():
                            multiview_steps = gr.Slider(minimum=10, maximum=50, value=30, step=5,
                                                       label='Multiview Quality Steps',
                                                       info='How many refinement steps for each texture view. More = higher quality textures but slower. 20-30 is usually enough, 40+ for best quality.')
                        gr.Markdown("**Shadow Removal** - Removes lighting/shadows baked into your input image for cleaner textures")
                        with gr.Row():
                            delight_steps = gr.Slider(minimum=20, maximum=80, value=50, step=5,
                                                     label='Shadow Removal Steps',
                                                     info='How thoroughly to remove shadows from your input. More steps = cleaner removal but slower. 40-60 is usually good.')
                        with gr.Row():
                            delight_cfg_image = gr.Slider(minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                                                         label='Structure Preservation',
                                                         info='How much to keep the original image details. Higher = preserves more detail but may keep some shadows. Lower = more aggressive shadow removal but may lose details.')
                            delight_cfg_text = gr.Slider(minimum=0.5, maximum=3.0, value=1.0, step=0.1,
                                                        label='Shadow Removal Strength',
                                                        info='How strongly to apply shadow removal. Increase if shadows remain visible. Decrease if the result looks too flat or washed out.')
                        with gr.Row():
                            use_antialias = gr.Checkbox(value=True, label='Smooth Edges (Antialiasing)',
                                                       info='Smooths jagged/pixelated edges in the texture. Leave ON for better quality. Only turn OFF for slightly faster processing.')
                    with gr.Tab('Mesh Options', id='tab_mesh_options'):
                        gr.Markdown("**Cleanup** - Removes floating bits and debris from your 3D model")
                        floater_ratio = gr.Slider(minimum=0.001, maximum=0.05, value=0.005, step=0.001,
                                                 label='Floater Cleanup Threshold',
                                                 info='Removes small disconnected pieces (like floating dots or fragments). Higher values remove more aggressively - be careful not to remove intentional small parts like buttons or eyes.')
                        gr.Markdown("**Mesh Simplification** - Reduces polygon count while keeping the shape looking good")
                        with gr.Row():
                            quality_thr = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.1,
                                                   label='Quality Priority',
                                                   info='Balance between quality and polygon reduction. 1.0 = prioritize quality (may keep more polygons). Lower = more aggressive reduction (smaller file but may look blockier).')
                            preserve_boundary = gr.Checkbox(value=True, label='Protect Edges',
                                                           info='Keeps the outline/silhouette of your model sharp. Turn OFF only if you need maximum polygon reduction and don\'t mind softer edges.')
                        with gr.Row():
                            boundary_weight = gr.Slider(minimum=1, maximum=10, value=3, step=1,
                                                       label='Edge Protection Strength',
                                                       info='How strongly to protect edges when simplifying (only works if Protect Edges is ON). Higher = edges are more protected. 3 is a good balance.')
                            preserve_normal = gr.Checkbox(value=True, label='Keep Smooth Shading',
                                                         info='Maintains the smooth appearance of curved surfaces. Turn OFF for more aggressive reduction, but the model may look more faceted/angular.')
                        with gr.Row():
                            preserve_topology = gr.Checkbox(value=True, label='Prevent Holes',
                                                           info='Prevents the simplification from creating holes or merging separate parts. IMPORTANT: Keep ON if you plan to 3D print! Turn OFF only for maximum reduction.')
                            auto_clean = gr.Checkbox(value=True, label='Auto-Fix Errors',
                                                    info='Automatically fixes common mesh problems like duplicate points and malformed triangles. Leave ON unless you have a specific reason to disable it.')
                    with gr.Tab('Model Config', id='tab_model_config'):
                        gr.Markdown(f"""
## Current Model Configuration

| Setting | Value |
|---------|-------|
| Shape Model | `{args.model_path}/{args.subfolder}` |
| Texture Model | `{args.texgen_model_path}` |
| Turbo Mode | {'**Enabled** (faster, uses FlashVDM)' if TURBO_MODE else 'Disabled'} |
| Text-to-3D | {'**Enabled** (HunyuanDiT)' if HAS_T2I else 'Disabled'} |
| Memory Profile | {args.profile} |

---

## Available Models

| Flag | Model | Size | Description |
|------|-------|------|-------------|
| (default) | Hunyuan3D-2mini | 0.6B | Fast, good quality. Best for most uses. |
| `--h2` | Hunyuan3D-2 | 1.1B | Full model. Higher quality, especially for complex shapes. |
| `--mv` | Hunyuan3D-2mv | 1.1B | Multiview input. Use when you have multiple view images. |
| `--turbo` | +Turbo | - | Add to any model for 5-step fast generation. Slight quality tradeoff. |

## Memory Profiles

| Profile | RAM | VRAM | Speed | Use When |
|---------|-----|------|-------|----------|
| 1 | High | High | Fastest | 32GB+ RAM, 12GB+ VRAM |
| 2 | High | Low | Fast | 32GB+ RAM, 6-8GB VRAM |
| 3 | Low | High | Medium | 16GB RAM, 12GB+ VRAM |
| 4 | Low | Low | Slower | 16GB RAM, 6-8GB VRAM |
| 5 | Very Low | Low | Slowest | 8GB RAM, 6GB VRAM |

---

## Restart Commands

**Changing models requires a restart.** Example commands:

```
# Mini model (default, fast)
python gradio_app.py --profile 3 --enable_t23d

# Full model (higher quality)
python gradio_app.py --h2 --profile 3 --enable_t23d

# Turbo mode (5-step fast generation)
python gradio_app.py --turbo --profile 3 --enable_t23d

# Multiview input model
python gradio_app.py --mv --profile 3
```
                        """)
                    with gr.Tab("Export", id='tab_export'):
                        gr.Markdown("**Export Settings** - Configure how to save your 3D model")
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100,
                                                    info='GLB = best for web/games (includes textures). OBJ = widely compatible. FBX = good for animation software. STL = for 3D printing.')
                            reduce_face = gr.Checkbox(label='Simplify Mesh', value=False, min_width=100,
                                                     info='Reduces polygon count to make a smaller file. Useful for web/games or if the model is too detailed.')
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100,
                                                         info='Include the color texture in the export. Only available for textured models.')
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Polygon Count',
                                                    info='Number of triangles in the simplified mesh. Lower = smaller file/faster loading. 10,000 is good for web. 50,000+ for high-quality renders.')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=True,
                                        min_width=100)

            with gr.Column(scale=4, elem_id="tabs_output"):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label=None, examples_per_page=18)
                    with gr.Tab('MultiView to 3D Gallery', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label=None, examples_per_page=100)

        gr.HTML(f"""
        <div align="center" style="color: #64748b; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem; font-size: 0.8rem;">
            Archeon 3D Engine &bull; Shape: {args.model_path}/{args.subfolder} &bull; Texture: {'Disabled' if args.disable_tex else 'Vanguard-H3D (Ready)'}
            <br>
            <span style="opacity: 0.5;">Based on Tencent Hunyuan3D-2.0 | Archeon Core Infrastructure</span>
        </div>
        """)

        # Warnings removed for cleaned Archeon UI

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)
        tab_mv.select(fn=lambda: gr.update(selected='tab_mv_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                # Shape options
                mc_level,
                bounds,
                eta,
                min_resolution,
                s_churn,
                s_noise,
                s_tmin,
                s_tmax,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                # Shape options
                mc_level,
                bounds,
                eta,
                min_resolution,
                s_churn,
                s_noise,
                s_tmin,
                s_tmax,
                # Texture options
                texture_size,
                render_size,
                bake_exp,
                bake_angle_thres,
                multiview_steps,
                delight_steps,
                delight_cfg_image,
                delight_cfg_text,
                use_antialias,
                # Mesh options
                floater_ratio,
                quality_thr,
                preserve_boundary,
                boundary_weight,
                preserve_normal,
                preserve_topology,
                auto_clean,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                             textured=True)
            else:
                mesh = trimesh.load(file_out)
                floater_remove, degenerate_remove, face_reduce = get_postprocessors()
                mesh = floater_remove(mesh)
                mesh = degenerate_remove(mesh)
                if reduce_face:
                    mesh = face_reduce(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH,
                                                             textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )


    return demo


def replace_property_getter(instance, property_name, new_getter):
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)
    instance.__class__ = custom_class
    return instance


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='dmc')
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(_XDG_CACHE, 'hy3dgen', 'launcher'))
    parser.add_argument('--enable_t23d', action='store_true', default=True)
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")

    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--low-vram-mode', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--turbo', action='store_true')
    parser.add_argument('--mv', action='store_true')
    parser.add_argument('--h2', action='store_true')

    global args, SAVE_DIR, CURRENT_DIR, MV_MODE, TURBO_MODE, HTML_HEIGHT, HTML_WIDTH, \
        HTML_OUTPUT_PLACEHOLDER, INPUT_MESH_HTML, example_is, example_ts, example_mvs, SUPPORTED_FORMATS, \
        HAS_TEXTUREGEN, texgen_worker, rmbg_worker, i23d_worker, floater_remove_worker, \
        degenerate_face_remove_worker, face_reduce_worker, t2i_worker, HAS_T2I, \
        export_to_trimesh, FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
        Hunyuan3DDiTFlowMatchingPipeline, BackgroundRemover

    args = parser.parse_args()

    if args.mini:
        args.model_path = "tencent/Hunyuan3D-2mini"
        args.subfolder = "hunyuan3d-dit-v2-mini"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.mv:
        args.model_path = "tencent/Hunyuan3D-2mv"
        args.subfolder = "hunyuan3d-dit-v2-mv"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.h2:
        args.model_path = "tencent/Hunyuan3D-2"
        args.subfolder = "hunyuan3d-dit-v2-0"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.turbo:
        args.subfolder = args.subfolder + "-turbo"
        args.enable_flashvdm = True

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        from hy3dgen.version import check_for_updates
        update_info = check_for_updates()
        if update_info:
            logger.info(f"Update available: {update_info['latest']} → {update_info['url']}")
    except Exception:
        pass

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 635
    HTML_WIDTH = 1000

    HTML_OUTPUT_PLACEHOLDER = f'''
    <div style='height: {HTML_HEIGHT}px; width: 100%; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    '''

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    torch.set_default_device("cpu")
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    # --- Fast Initialization ---
    app = FastAPI()

    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "model": f"{args.model_path}/{args.subfolder}",
            "texture_loaded": texgen_worker is not None,
            "shape_loaded": i23d_worker is not None,
            "text2image_enabled": args.enable_t23d,
        }

    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    launcher_app = gr.mount_gradio_app(app, demo, path="/")

    def open_browser():
        target_url = f"http://{args.host}:{args.port}"
        logger.info(f"Opening browser at {target_url}")
        webbrowser.open_new_tab(target_url)

    Timer(1.5, open_browser).start()
    uvicorn.run(launcher_app, host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
