import sys
import os
import traceback

print("Python executable:", sys.executable)
print("CWD:", os.getcwd())

try:
    print("Attempting to import Hunyuan3DPaintPipeline...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    print("Import successful.")

    model_path = 'tencent/Hunyuan3D-2'
    print(f"Attempting to load from_pretrained('{model_path}')...")
    
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
    print("Load successful!")

except Exception:
    print("FAILURE!")
    traceback.print_exc()
