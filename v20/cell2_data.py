import os, re, gc, glob, math, random, shutil, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback
torch.backends.cudnn.benchmark = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============ CONFIG ============
B = "/content/drive/MyDrive/Colab Notebooks/My Drive/akkadian"
MP = B + "/models/byt5-akkadian-modelpart1"
DD = B + "/data"
O = B + "/output/byt5-v20"
os.makedirs(O, exist_ok=True)

RESUME = None  # Set to "ph1" or "ph2" to skip completed phases
PFX = "translate Akkadian to English: "
MS = 512
MT = 512
BA = 1
GA = 32
P1E = 3
P2E = 1
P3E = 8
AN = 3
RP = 0.15
VN = 200
DV = "cuda"

# ============ CHECKS ============
assert torch.cuda.is_available(), "No GPU"
GP = torch.cuda.get_device_properties(0)

def get_vram():
    g = torch.cuda.get_device_properties(0)
    return g.total_memory if hasattr(g, "total_memory") else getattr(g, "total_mem", 0)

assert get_vram() > 14e9, "Need 15GB+"
assert os.path.isfile(MP + "/config.json"), "Model missing: " + MP
TP = MP
if not os.path.isfile(TP + "/tokenizer_config.json"):
    TP = B + "/models/byt5-akkadian-model_final"
assert os.path.isfile(DD + "/train.csv"), "train.csv missing"
assert os.path.isfile(DD + "/test.csv"), "test.csv missing"
print("GPU=" + GP.name + " VRAM=" + str(round(get_vram() / 1e9, 1)) + "GB")

# ============ HELPERS ============
def fr():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def gfr():
    return (get_vram() - torch.cuda.memory_allocated(0)) / 1e9

_TR = str.maketrans(
    "\u0101\u0113\u012b\u016b\u0100\u0112\u012a\u016a\u00e1\u00e9\u00ed\u00fa\u00e0\u00e8\u00ec\u00f9\u00c1\u00c9\u00cd\u00da\u00c0\u00c8\u00cc\u00d9\u0161\u0160\u1e63\u1e62\u1e6d\u1e6c\u1e2b\u1e2a",
    "aeiuAEIUaeiuaeiuAEIUAEIUsSsStThH")

def nk(t):
    return re.sub(r"\s+", " ", str(t).strip().lower().translate(_TR))

def ng(t):
    t = re.sub(r"<gap>", "...", str(t))
    t = re.sub(r"\[\.+\]", "...", t)
    return t.strip()

# ============ AUGMENTATION ============
DM = {}
DM["\u0101"] = "a"
DM["\u0113"] = "e"
DM["\u012b"] = "i"
DM["\u016b"] = "u"
DM["\u0100"] = "A"
DM["\u0112"] = "E"
DM["\u012a"] = "I"
DM["\u016a"] = "U"
DM["\u0161"] = "sh"
DM["\u0160"] = "SH"
DM["\u1e63"] = "s"
DM["\u1e62"] = "S"
DM["\u1e6d"] = "t"
DM["\u1e6c"] = "T"
DM["\u1e2b"] = "h"
DM["\u1e2a"] = "H"
DM["\u02be"] = "'"
DM["\u02bf"] = "'"

def _ad(t, p=0.25):
    out = ""
    for c in str(t):
        if c in DM and random.random() < p:
            out = out + DM[c]
        else:
            out = out + c
    return out

def _sw(t, p=0.06):
    c = list(str(t))
    for i in range(len(c) - 1):
        if random.random() < p:
            c[i], c[i + 1] = c[i + 1], c[i]
    return "".join(c)

def _ag(t, p=0.10):
    w = str(t).split()
    r = []
    for x in w:
        r.append(x)
        if random.random() < p and len(r) < len(w):
            r.append("...")
    return " ".join(r)

def augment(df, n=AN):
    rows = []
    fns = [_ad, _sw, _ag]
    for _ in range(n):
        for _, row in df.iterrows():
            s = str(row["transliteration"])
            for fn in random.sample(fns, k=random.randint(1, 3)):
                s = fn(s)
            rows.append({
                "transliteration": s,
                "translation": str(row["translation"])
            })
    out = pd.DataFrame(rows)
    orig = set(df["transliteration"].apply(nk))
    return out[~out["transliteration"].apply(nk).isin(orig)].reset_index(drop=True)

# ============ LOAD DATA ============
print("Loading data...")

gold = pd.read_csv(DD + "/train.csv")[["transliteration", "translation"]].dropna()
gold = gold[gold["transliteration"].str.len() > 3]
gold = gold[gold["translation"].str.len() > 3].reset_index(drop=True)
test_df = pd.read_csv(DD + "/test.csv")
tkeys = set(test_df["transliteration"].astype(str).apply(nk))
gold = gold[~gold["transliteration"].apply(nk).isin(tkeys)].reset_index(drop=True)
print("Gold: " + str(len(gold)))

def load_sv(path):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["transliteration", "translation"])
    r = pd.read_csv(path)[["transliteration", "translation"]].dropna()
    r = r[r["transliteration"].str.len() > 5]
    r = r[r["translation"].str.len() > 5]
    r = r[~r["translation"].str.contains(r"https?://", regex=True, na=False)]
    r = r[r["translation"].str.contains(r"[a-zA-Z]", na=False)]
    ratio = r["translation"].str.len() / (r["transliteration"].str.len() + 1)
    return r[(ratio >= 0.3) & (ratio <= 5.0)].reset_index(drop=True)

cdli = load_sv(DD + "/cdli_pseudo_labels.csv")
gem = load_sv(DD + "/gemini_pseudo_labeled_train.csv")
bt = load_sv(DD + "/gemini_back_translated_train.csv")
print("CDLI:" + str(len(cdli)) + " Gem:" + str(len(gem)) + " BT:" + str(len(bt)))

parts = [d for d in [cdli, gem] if len(d) > 0]
if parts:
    silver = pd.concat(parts, ignore_index=True)
else:
    silver = pd.DataFrame(columns=["transliteration", "translation"])
silver = silver[~silver["transliteration"].apply(nk).isin(tkeys)]
gkeys = set(gold["transliteration"].apply(nk))
silver = silver[~silver["transliteration"].apply(nk).isin(gkeys)].reset_index(drop=True)
if len(bt) > 0:
    bt = bt[~bt["transliteration"].apply(nk).isin(tkeys)]
    bt = bt[~bt["transliteration"].apply(nk).isin(gkeys)].reset_index(drop=True)
print("Silver:" + str(len(silver)) + " BT:" + str(len(bt)))

# ============ PHASE DATASETS ============
print("Building phases...")

val = gold.sample(min(VN, len(gold)), random_state=42)
vkeys = set(val["transliteration"].apply(nk))

def ev(df):
    if len(df) == 0:
        return df
    return df[~df["transliteration"].apply(nk).isin(vkeys)].reset_index(drop=True)

gt = ev(gold)
ga = augment(gt)
bte = ev(bt) if len(bt) > 0 else pd.DataFrame()
print("GoldTr:" + str(len(gt)) + " Aug:" + str(len(ga)) + " BT:" + str(len(bte)))

p1_parts = [d for d in [gt, ga, bte] if len(d) > 0]
ph1 = pd.concat(p1_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

se = ev(silver)
rn = int(len(se) * RP / max(1 - RP, 0.01))
rep = gt.sample(min(rn, len(gt)), random_state=7)
p2_parts = [d for d in [se, rep] if len(d) > 0]
if p2_parts:
    ph2 = pd.concat(p2_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
else:
    ph2 = pd.DataFrame()

ph3 = pd.concat([gt, ga], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

spe3 = math.ceil(len(ph3) / GA)
total_h = (math.ceil(len(ph1) / GA) * P1E + math.ceil(len(ph2) / GA) * P2E + spe3 * P3E) / 0.08 / 3600
while total_h > 10 and P3E > 5:
    P3E -= 1
    total_h = (math.ceil(len(ph1) / GA) * P1E + math.ceil(len(ph2) / GA) * P2E + spe3 * P3E) / 0.08 / 3600

print("Ph1:" + str(len(ph1)) + "x" + str(P1E) + "ep Ph2:" + str(len(ph2)) + "x" + str(P2E) + "ep Ph3:" + str(len(ph3)) + "x" + str(P3E) + "ep")
print("Estimated: " + str(round(total_h, 1)) + "h")

for nm, df in [("ph1", ph1), ("ph2", ph2), ("ph3", ph3), ("val", val)]:
    if len(df) == 0:
        continue
    leaked = set(df["transliteration"].apply(nk)) & tkeys
    assert not leaked, "LEAK in " + nm
print("Zero leakage confirmed")
print("Data setup complete")
