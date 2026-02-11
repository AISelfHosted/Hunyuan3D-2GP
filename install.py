import os
import sys
import subprocess
import venv
from pathlib import Path

# Configuration
VENV_DIR = Path(".venv")
PIP_CMD = [str(VENV_DIR / "bin" / "pip")]
PYTHON_CMD = [str(VENV_DIR / "bin" / "python")]

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR}...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"Using existing virtual environment in {VENV_DIR}.")

def install_package():
    """Install the package and its dependencies using pip from the venv."""
    print("Installing dependencies and building package...")
    
    try:
        # Upgrade pip first
        subprocess.check_call(PIP_CMD + ["install", "--upgrade", "pip"])
        
        # Install in editable mode to trigger setup.py
        # This will install requirements.txt and compile custom_rasterizer
        subprocess.check_call(PIP_CMD + ["install", "-e", "."])
        
        print("\n✅ Installation complete!")
        print(f"To activate the environment, run: source {VENV_DIR}/bin/activate")
        print("Then you can run the app with: hy3dgen-gradio")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed with error code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ Unexpected error during installation: {e}")
        sys.exit(1)

def main():
    print("=== Hunyuan3D-2GP Auto-Installer ===")
    
    # 1. Create/Verify Virtual Environment
    create_venv()
    
    # 2. Install Package
    install_package()

if __name__ == "__main__":
    main()
