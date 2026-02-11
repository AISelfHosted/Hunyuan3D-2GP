import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import pytest

# Mocking modules that might fail without GPU or specific deps
with patch.dict('sys.modules', {'hy3dgen.texgen.differentiable_renderer.mesh_render': MagicMock()}):
    try:
        from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline
    except ImportError:
        # Fallback if patch fails - likely environment has deps installed
        from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline

class TestTexGenLoading:
    @pytest.fixture
    def mock_home(self):
        tmp_dir = tempfile.mkdtemp()
        yield tmp_dir
        shutil.rmtree(tmp_dir)

    def test_load_from_local_path(self, mock_home):
        # Create fake model structure
        model_path = os.path.join(mock_home, 'my-local-model')
        os.makedirs(os.path.join(model_path, 'hunyuan3d-delight-v2-0'))
        os.makedirs(os.path.join(model_path, 'hunyuan3d-paint-v2-0'))

        # Mock the __init__ to avoid actually loading heavy models
        with patch.object(Hunyuan3DPaintPipeline, '__init__', return_value=None) as mock_init:
            pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            
            # Verify it called init with correct config paths
            args, _ = mock_init.call_args
            config = args[0]
            assert config.light_remover_ckpt_path == os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            assert config.multiview_ckpt_path == os.path.join(model_path, 'hunyuan3d-paint-v2-0')

    def test_load_from_cache(self, mock_home):
        # Setup fake cache structure
        repo_id = "tencent/Hunyuan3D-2"
        # Since we use os.path.join in the code:
        # base_dir = os.environ.get('HY3DGEN_MODELS', os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'hy3dgen'))
        # cached_model_path = os.path.expanduser(os.path.join(base_dir, model_path))
        
        # So if XDG_CACHE_HOME is set to mock_home/.cache
        xdg_cache = os.path.join(mock_home, ".cache")
        hy3dgen_cache = os.path.join(xdg_cache, "hy3dgen")
        cached_model_path = os.path.join(hy3dgen_cache, repo_id)
        
        os.makedirs(os.path.join(cached_model_path, 'hunyuan3d-delight-v2-0'), exist_ok=True)
        os.makedirs(os.path.join(cached_model_path, 'hunyuan3d-paint-v2-0'), exist_ok=True)

        with patch.dict(os.environ, {'XDG_CACHE_HOME': xdg_cache}):
             with patch.object(Hunyuan3DPaintPipeline, '__init__', return_value=None) as mock_init:
                pipeline = Hunyuan3DPaintPipeline.from_pretrained(repo_id)
                
                args, _ = mock_init.call_args
                config = args[0]
                assert config.light_remover_ckpt_path == os.path.join(cached_model_path, 'hunyuan3d-delight-v2-0')

    def test_download_triggered_if_missing(self, mock_home):
        repo_id = "tencent/Hunyuan3D-2-missing"
        xdg_cache = os.path.join(mock_home, ".cache")
        
        # Mock huggingface_hub
        # Note: We patch where it is imported in the module, or sys.modules if inside function
        # Since it is imported inside the method in pipelines.py, patch('hy3dgen.texgen.pipelines.huggingface_hub') won't work easily if it's not at top level
        # Wait, the code does `import huggingface_hub` inside the method.
        # So we need to patch sys.modules['huggingface_hub'] BEFORE calling the method.
        
        mock_hf = MagicMock()
        def side_effect(repo_id, allow_patterns, local_dir, local_dir_use_symlinks=True):
            # Create the dirs to simulate successful download
            if "delight" in allow_patterns[0]:
                os.makedirs(os.path.join(local_dir, 'hunyuan3d-delight-v2-0'), exist_ok=True)
            elif "paint" in allow_patterns[0]:
                os.makedirs(os.path.join(local_dir, 'hunyuan3d-paint-v2-0'), exist_ok=True)
            return local_dir

        mock_hf.snapshot_download.side_effect = side_effect
        
        with patch.dict(os.environ, {'XDG_CACHE_HOME': xdg_cache}):
             with patch.dict('sys.modules', {'huggingface_hub': mock_hf}):
                with patch.object(Hunyuan3DPaintPipeline, '__init__', return_value=None) as mock_init:
                    pipeline = Hunyuan3DPaintPipeline.from_pretrained(repo_id)
                    
                    assert mock_hf.snapshot_download.call_count == 2
