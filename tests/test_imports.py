"""
Smoke tests for Hunyuan3D-2GP.
Tests that all critical modules can be imported and basic classes exist.
"""
import sys
import os
import importlib


def test_hy3dgen_import():
    """Test that the core hy3dgen package imports without error."""
    import hy3dgen
    assert hasattr(hy3dgen, '__file__')


def test_shapegen_imports():
    """Test shapegen module imports and key classes exist."""
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
        FaceReducer,
        MeshSimplifier,
    )
    assert Hunyuan3DDiTFlowMatchingPipeline is not None
    assert FloaterRemover is not None
    assert DegenerateFaceRemover is not None
    assert FaceReducer is not None
    assert MeshSimplifier is not None


def test_shapegen_pipelines_module():
    """Test that shapegen.pipelines and export_to_trimesh exist."""
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    assert callable(export_to_trimesh)


def test_rembg_import():
    """Test background remover imports."""
    from hy3dgen.rembg import BackgroundRemover
    assert BackgroundRemover is not None


def test_text2image_import():
    """Test text2image module imports."""
    from hy3dgen.text2image import HunyuanDiTPipeline
    assert HunyuanDiTPipeline is not None


def test_shapegen_utils():
    """Test that shapegen utils and logger exist."""
    from hy3dgen.shapegen.utils import logger
    assert logger is not None


def test_postprocessors_are_callable():
    """Test that postprocessor classes can be instantiated."""
    from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
    fr = FloaterRemover()
    dfr = DegenerateFaceRemover()
    face_r = FaceReducer()
    assert callable(fr)
    assert callable(dfr)
    assert callable(face_r)
