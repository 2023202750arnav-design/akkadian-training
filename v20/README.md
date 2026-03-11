# v20 Akkadian Translation Pipeline

Complete training + inference pipeline for the Deep Past Akkadian-to-English translation competition.

## Mode 1 — Recommended: Clone repo and run from Drive (2 Colab cells)

Copy-pasting scripts into Colab can corrupt indentation. Clone the repo instead
and run the files directly with `%run`.

**Cell 1** — setup (paste this once):

```python
from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys
for p in ["transformers", "datasets", "sacrebleu", "sentencepiece"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])

import transformers, torch
print("transformers=" + transformers.__version__ + " torch=" + torch.__version__)

if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    _vram = g.total_memory if hasattr(g, "total_memory") else getattr(g, "total_mem", 0)
    print("GPU=" + g.name + " VRAM=" + str(round(_vram / 1e9, 1)) + "GB")
else:
    print("NO GPU - set Runtime to T4")
```

**Cell 2** — clone repo and run (paste this once):

```python
import subprocess
subprocess.run(["git", "clone", "--depth=1",
    "https://github.com/2023202750arnav-design/akkadian-training.git",
    "/content/scripts"], check=True)

%run /content/scripts/v20/cell234_full.py
```

> **Note:** `cell234_full.py` merges data loading, training, and inference into
> one file so the entire pipeline runs in a single `%run` command.

---

## Mode 2 — Alternative: Copy-paste into 4 Colab cells

Use this only if you cannot clone the repo. Run the four scripts in order:

1. `v20/cell1_setup.py`
2. `v20/cell2_data.py`
3. `v20/cell3_train.py`
4. `v20/cell4_inference.py`

## Resume After Disconnect

If Colab disconnects during training, open `cell234_full.py` (or `cell2_data.py`
in Mode 2) and change:

```
RESUME = None
```

to:

```
RESUME = "ph1"   # if Phase 1 finished
RESUME = "ph2"   # if Phase 1 and 2 finished
```

Then rerun from Cell 2 (Mode 1) or from cell2 onward (Mode 2).

## Architecture

- **Phase 1** (~2h): Gold + augmented + back-translations warmup (3 epochs, lr=3e-5)
- **Phase 2** (~1h): Silver (CDLI + Gemini) + gold replay (1 epoch, lr=1e-5)
- **Phase 3** (~5h): Gold + augmented refinement with EMA (8 epochs, lr=5e-6)
- **Inference**: Diverse MBR decoding (beam + diverse beam + sampling → chrF++ MBR)

## Key Features

- EMA (Exponential Moving Average) for smoother weights
- Checkpoint averaging of last 3 checkpoints
- Loss explosion detector (prevents v19-style disasters)
- Score tracking with BLEU/chrF++/GeoMean each epoch
- Diverse MBR: ~46 candidates per row → consensus selection
- Two submissions: MBR (primary) + beam (backup)

## Expected Time

Total: ~8 hours on T4 GPU (fits 12h Colab session)
