from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys
for p in ["transformers", "datasets", "sacrebleu", "sentencepiece"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])

import transformers, torch
print("transformers=" + transformers.__version__ + " torch=" + torch.__version__)

def get_vram():
    g = torch.cuda.get_device_properties(0)
    return g.total_memory if hasattr(g, "total_memory") else getattr(g, "total_mem", 0)

if torch.cuda.is_available():
    print("GPU=" + torch.cuda.get_device_properties(0).name + " VRAM=" + str(round(get_vram() / 1e9, 1)) + "GB")
else:
    print("NO GPU - set Runtime to T4")
