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
    )
    path = export_mesh(mesh, save_folder, textured=False)

    tmp_time = time.time()
    floater_remove, degenerate_remove, face_reduce = get_postprocessors()
    mesh = face_reduce(mesh)
    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

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

    /* Glassmorphism effects */
    .gr-panel, .gr-block, .gr-box {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }

    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }
    
    /* Branding Header */
    .archeon-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    
    .archeon-badge {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* Remove Gradio Branding */
    footer {
        display: none !important;
    }
    
    .gr-button-secondary.api-link {
        display: none !important;
    }

    /* Force tabs to same row */
    .tabs > .tab-nav {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }
    
    .tabs > .tab-nav > button {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }
    """

    with gr.Blocks(theme=archeon_theme, title='Archeon 3D Launcher', analytics_enabled=False, css=custom_css) as demo:
        with gr.Column(elem_id="header-container"):
            gr.HTML(f"""
            <div style="text-align: center; margin-bottom: 0.5rem; margin-top: 0.5rem;">
                <h1 style="font-size: 1.8rem; margin-bottom: 0.1rem;" class="archeon-header">ARCHEON 3D <span class="archeon-badge">SIDE CAR READY</span></h1>
                <p style="color: #94a3b8; font-size: 0.9rem; max-width: 800px; margin: 0 auto;">
                    Professional 3D Generation Pipeline. Powered by Tencent Hunyuan3D-2.0 & GPU-Poor optimizations.
                </p>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=3):
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

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=True,
                                        min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(label='Generation Mode',
                                            info='Recommendation: Turbo for most cases, Fast for very complex cases, Standard seldom use.',
                                            choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                        decode_mode = gr.Radio(label='Decoding Mode',
                                               info='The resolution for exporting mesh from generated vectset',
                                               choices=['Low', 'Standard', 'High'],
                                               value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery') as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=9)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery') as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label=None, examples_per_page=9)
                    with gr.Tab('MultiView Gallery', id='tab_mv_gallery') as tab_mv_gal:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label=None, examples_per_page=4)

        gr.HTML(f"""
        <div align="center" style="color: #64748b; margin-top: 0.5rem; border-top: 1px solid #1f2937; padding-top: 0.5rem; font-size: 0.8rem;">
            Archeon 3D Engine &bull; Shape: {args.model_path}/{args.subfolder} &bull; Texture: {'Vanguard-H3D' if HAS_TEXTUREGEN else 'Disabled'}
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

    HTML_HEIGHT = 580
    HTML_WIDTH = 1000

    HTML_OUTPUT_PLACEHOLDER = f'''
    <div style='height: {HTML_HEIGHT}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
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
