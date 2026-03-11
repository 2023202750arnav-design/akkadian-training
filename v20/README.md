# v20 Akkadian Translation Pipeline

Complete training + inference pipeline for the Deep Past Akkadian-to-English translation competition.

## Quick Start (Google Colab)

1. Clone this repo in Colab:

```
!git clone https://github.com/2023202750arnav-design/akkadian-training.git /content/scripts
```

2. Run cells in order:

```
%run /content/scripts/v20/cell1_setup.py
%run /content/scripts/v20/cell2_data.py
%run /content/scripts/v20/cell3_train.py
%run /content/scripts/v20/cell4_inference.py
```

## Resume After Disconnect

If Colab disconnects during training, edit `cell2_data.py` and change:

```
RESUME = None
```

to:

```
RESUME = "ph1"   # if Phase 1 finished
RESUME = "ph2"   # if Phase 1 and 2 finished
```

Then rerun all cells.

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
