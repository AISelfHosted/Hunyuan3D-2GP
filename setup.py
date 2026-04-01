import os
import sys
from setuptools import setup, find_packages

def get_requirements():
    try:
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
        # Filter out comments and empty lines
        requirements = [r.strip() for r in requirements if r.strip() and not r.strip().startswith('#')]
        return requirements
    except FileNotFoundError:
        return []

# Custom Rasterizer Extension Build
def get_extensions():
    extensions = []
    
    # Check for torch and CUDA availability
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        
        if torch.cuda.is_available():
            extensions.append(
                CUDAExtension(
                    name='custom_rasterizer_kernel',
                    sources=[
                        'hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp',
                        'hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/grid_neighbor.cpp',
                        'hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
                    ],
                    extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
                )
            )
            return extensions, {'build_ext': BuildExtension}
        else:
            print("WARNING: CUDA not available. Skipping custom_rasterizer build.")
            return [], {}
            
    except ImportError:
        print("WARNING: torch not installed. Skipping custom_rasterizer build.")
        return [], {}
    except Exception as e:
        print(f"WARNING: Failed to configure custom_rasterizer build: {e}")
        return [], {}

ext_modules, cmdclass = get_extensions()

setup(
    name='hy3dgen',
    version='0.1.0',
    description='Hunyuan3D-2GP: Open Source 3D Generation Pipeline',
    packages=find_packages(),
    # Explicitly include custom_rasterizer package mapping if not found by find_packages due to directory structure
    package_dir={
        'hy3dgen': 'hy3dgen',
        'custom_rasterizer': 'hy3dgen/texgen/custom_rasterizer/custom_rasterizer'
    },
    # Ensure custom_rasterizer is treated as a top-level package
    py_modules=['custom_rasterizer'] if not os.path.exists('hy3dgen/texgen/custom_rasterizer/custom_rasterizer') else [],
    
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={
        'console_scripts': [
            'hy3dgen-api=hy3dgen.api.server:main',
            'hy3dgen-launcher=launcher:main',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True,
)
