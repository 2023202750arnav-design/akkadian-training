from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys
for p in ["transformers", "datasets", "sacrebleu", "sentencepiece"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])

import transformers, torch
print("transformers=" + transformers.__version__ + " torch=" + torch.__version__)
if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    print("GPU=" + g.name + " VRAM=" + str(round(g.total_mem / 1e9, 1)) + "GB")
else:
    print("NO GPU - set Runtime to T4")
