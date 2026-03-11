# ============================================================
# CELL 2+3+4 MERGED: Data, Training, Inference
# ============================================================
import os, re, gc, glob, math, random, shutil, time, inspect
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

RESUME = None
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
    _g = torch.cuda.get_device_properties(0)
    return _g.total_memory if hasattr(_g, "total_memory") else getattr(_g, "total_mem", 0)

TOTAL_VRAM = get_vram()
assert TOTAL_VRAM > 14e9, "Need 15GB+"
assert os.path.isfile(MP + "/config.json"), "Model missing: " + MP
TP = MP
if not os.path.isfile(TP + "/tokenizer_config.json"):
    TP = B + "/models/byt5-akkadian-model_final"
assert os.path.isfile(DD + "/train.csv"), "train.csv missing"
assert os.path.isfile(DD + "/test.csv"), "test.csv missing"
print("GPU=" + GP.name + " VRAM=" + str(round(TOTAL_VRAM / 1e9, 1)) + "GB")

# ============ HELPERS ============
def fr():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def gfr():
    return (TOTAL_VRAM - torch.cuda.memory_allocated(0)) / 1e9

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

# ============================================================
# CELL 3: TRAINING
# ============================================================

class EMACallback(TrainerCallback):
    def __init__(self, mdl, decay=0.999):
        self.decay = decay
        self.steps = 0
        self.ema = {}
        for k, v in mdl.state_dict().items():
            self.ema[k] = v.detach().float().cpu().clone()
        print("EMA init: " + str(len(self.ema)) + " params on CPU")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        self.steps += 1
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.ema:
                    self.ema[k].mul_(d).add_(v.float().cpu(), alpha=1 - d)

    def save_ema(self, model, path):
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        backup = {}
        for k, v in model.state_dict().items():
            backup[k] = v.clone()
        ema_sd = {}
        for k, v in self.ema.items():
            ema_sd[k] = v.to(device=device, dtype=dtype)
        model.load_state_dict(ema_sd)
        model.save_pretrained(path)
        model.load_state_dict(backup)
        del backup, ema_sd
        fr()
        print("EMA saved to " + path + " (" + str(self.steps) + " steps)")


class LossChecker(TrainerCallback):
    def __init__(self, mx=50.0):
        self.mx = mx
        self.checked = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.checked:
            return
        loss = logs.get("loss")
        if loss is not None:
            if not math.isfinite(loss) or loss > self.mx:
                print("LOSS EXPLOSION: " + str(loss) + " - STOPPING")
                control.should_training_stop = True
            else:
                print("Loss OK: " + str(round(loss, 4)))
                self.checked = True


class ScoreTracker(TrainerCallback):
    def __init__(self, val_df, tokenizer, device, n=50):
        self.sample = val_df.sample(
            min(n, len(val_df)), random_state=99
        ).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = device
        self.best = 0.0
        self.history = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        was_train = model.training
        model.eval()
        model.config.use_cache = True
        preds = []
        refs = []
        with torch.no_grad():
            for _, row in self.sample.iterrows():
                enc = self.tokenizer(
                    PFX + str(row["transliteration"]),
                    max_length=MS,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                out = model.generate(
                    **enc,
                    max_new_tokens=256,
                    num_beams=4,
                    early_stopping=True
                )
                pred = ng(self.tokenizer.decode(out[0], skip_special_tokens=True))
                preds.append(pred)
                refs.append(ng(str(row["translation"])))
        model.config.use_cache = False
        try:
            from sacrebleu.metrics import BLEU, CHRF
            b = BLEU(effective_order=True).corpus_score(preds, [refs]).score
            c = CHRF(word_order=2).corpus_score(preds, [refs]).score
            s = math.sqrt(max(b * c, 0))
            flag = " BEST" if s > self.best else ""
            if s > self.best:
                self.best = s
            self.history.append((int(state.epoch), b, c, s))
            print(
                "ep" + str(int(state.epoch))
                + " BLEU=" + str(round(b, 2))
                + " chrF++=" + str(round(c, 2))
                + " Score=" + str(round(s, 2))
                + flag
            )
            if len(preds) > 0:
                print("P: " + preds[0][:90])
                print("R: " + refs[0][:90])
        except Exception as e:
            print("Score error: " + str(e))
        finally:
            if was_train:
                model.train()
            fr()


# ============ LOAD MODEL ============
print("Loading model...")
fr()

if RESUME == "ph2" and os.path.isfile(O + "/after_ph2/config.json"):
    load_path = O + "/after_ph2"
elif RESUME == "ph1" and os.path.isfile(O + "/after_ph1/config.json"):
    load_path = O + "/after_ph1"
else:
    load_path = MP
print("From: " + load_path)

try:
    tok = AutoTokenizer.from_pretrained(TP, use_fast=False)
except Exception:
    TP = B + "/models/byt5-akkadian-model_final"
    tok = AutoTokenizer.from_pretrained(TP, use_fast=False)
    print("Tokenizer fallback to model_final")

test_ids = tok(PFX + "test input", return_tensors="pt")
assert len(test_ids.input_ids[0]) > 3, "Tokenizer broken"
print("Tokenizer OK: " + type(tok).__name__)

mdl = AutoModelForSeq2SeqLM.from_pretrained(load_path, low_cpu_mem_usage=True).to(DV)
print("Params: " + str(sum(p.numel() for p in mdl.parameters())))
try:
    mdl.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
except TypeError:
    mdl.gradient_checkpointing_enable()
mdl.config.use_cache = False
print("Model loaded. GPU free: " + str(round(gfr(), 1)) + "GB")

# ============ TOKENIZE ============
print("Tokenizing...")

def tok_fn(ex):
    enc = tok(
        [PFX + str(s) for s in ex["transliteration"]],
        max_length=MS,
        truncation=True,
        padding=False,
    )
    lab = tok(
        [str(t) for t in ex["translation"]],
        max_length=MT,
        truncation=True,
        padding=False,
    )
    enc["labels"] = lab["input_ids"]
    return enc

def to_hf(df, name):
    if df is None or len(df) == 0:
        return None
    ds = HFDataset.from_pandas(
        df[["transliteration", "translation"]].reset_index(drop=True)
    ).map(
        tok_fn,
        batched=True,
        batch_size=64,
        remove_columns=["transliteration", "translation"],
        desc=name,
    )
    print(name + ": " + str(len(ds)))
    return ds

vt = to_hf(val, "val")
t1 = to_hf(ph1, "ph1")
t2 = to_hf(ph2, "ph2") if len(ph2) > 0 else None
t3 = to_hf(ph3, "ph3")
coll = DataCollatorForSeq2Seq(
    tokenizer=tok,
    model=mdl,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)
fr()

# ============ TRAINING ============
print("Starting training...")
scorer = ScoreTracker(val, tok, DV, n=50)
ema_cb = None

_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
_use_processing_class = "processing_class" in _trainer_sig

def make_args(sub, ep, lr, warm, smooth, save_steps=None):
    d = dict(
        output_dir=O + "/" + sub,
        num_train_epochs=ep,
        per_device_train_batch_size=BA,
        per_device_eval_batch_size=BA,
        gradient_accumulation_steps=GA,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warm,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        label_smoothing_factor=smooth,
        predict_with_generate=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        logging_steps=25,
        report_to="none",
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    try:
        d["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
        if save_steps:
            return Seq2SeqTrainingArguments(
                **d,
                eval_strategy="steps",
                eval_steps=save_steps,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=5,
            )
        return Seq2SeqTrainingArguments(
            **d,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
        )
    except TypeError:
        d.pop("gradient_checkpointing_kwargs", None)
        if save_steps:
            return Seq2SeqTrainingArguments(
                **d,
                eval_strategy="steps",
                eval_steps=save_steps,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=5,
            )
        return Seq2SeqTrainingArguments(
            **d,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
        )

def run_phase(label, ds, ep, lr, warm, smooth, sub, save_steps=None, use_ema=False):
    global ema_cb
    if ds is None or len(ds) == 0:
        print(label + ": SKIP (no data)")
        return
    print("")
    print("=" * 50)
    print(label)
    print("rows=" + str(len(ds)) + " ep=" + str(ep) + " lr=" + str(lr))
    print("=" * 50)
    t0 = time.time()
    cbs = [
        EarlyStoppingCallback(early_stopping_patience=max(3, ep + 2)),
        scorer,
        LossChecker(),
    ]
    if use_ema:
        ema_cb = EMACallback(mdl, decay=0.999)
        cbs.append(ema_cb)
        print("EMA: ON")
    trainer_kwargs = dict(
        model=mdl,
        args=make_args(sub, ep, lr, warm, smooth, save_steps),
        train_dataset=ds,
        eval_dataset=vt,
        data_collator=coll,
        callbacks=cbs,
    )
    if _use_processing_class:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()
    save_path = O + "/after_" + sub
    mdl.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    elapsed = round((time.time() - t0) / 3600, 1)
    print("Saved: " + save_path + " (" + str(elapsed) + "h)")
    if sub != "ph3":
        for c in glob.glob(O + "/" + sub + "/checkpoint-*"):
            try:
                shutil.rmtree(c)
            except Exception:
                pass
    fr()
    print("GPU free: " + str(round(gfr(), 1)) + "GB")

total_start = time.time()

if RESUME not in ("ph1", "ph2"):
    run_phase("PHASE 1: Gold+Aug+BT warmup", t1, P1E, 3e-5, 30, 0.10, "ph1")
else:
    print("Ph1: SKIPPED (resume)")

if RESUME != "ph2":
    run_phase("PHASE 2: Silver+Replay", t2, P2E, 1e-5, 50, 0.15, "ph2", save_steps=300)
else:
    print("Ph2: SKIPPED (resume)")

ph3_save = max(50, spe3)
run_phase(
    "PHASE 3: Gold+Aug refinement (EMA)",
    t3, P3E, 5e-6, 20, 0.05, "ph3",
    save_steps=ph3_save,
    use_ema=True,
)

total_elapsed = round((time.time() - total_start) / 3600, 1)
print("")
print("Training complete: " + str(total_elapsed) + "h")

# ============ CHECKPOINT AVERAGING ============
print("Checkpoint averaging...")
ckpts = sorted(glob.glob(O + "/ph3/checkpoint-*"))
if len(ckpts) >= 2:
    n = min(len(ckpts), 3)
    use_ckpts = ckpts[-n:]
    print("Averaging last " + str(n) + " checkpoints")
    avg_sd = None
    dt = next(mdl.parameters()).dtype
    for cp in use_ckpts:
        print("  " + os.path.basename(cp))
        m2 = AutoModelForSeq2SeqLM.from_pretrained(cp, low_cpu_mem_usage=True)
        sd = {}
        for k, v in m2.state_dict().items():
            sd[k] = v.float()
        if avg_sd is None:
            avg_sd = {}
            for k, v in sd.items():
                avg_sd[k] = v / n
        else:
            for k in avg_sd:
                avg_sd[k] = avg_sd[k] + sd[k] / n
        del m2, sd
        fr()
    avg_path = O + "/ckpt_avg"
    os.makedirs(avg_path, exist_ok=True)
    load_sd = {}
    for k, v in avg_sd.items():
        load_sd[k] = v.to(device=DV, dtype=dt)
    mdl.load_state_dict(load_sd)
    mdl.save_pretrained(avg_path)
    tok.save_pretrained(avg_path)
    print("Checkpoint avg saved: " + avg_path)
    del avg_sd, load_sd
    fr()
    bm = AutoModelForSeq2SeqLM.from_pretrained(
        O + "/after_ph3", low_cpu_mem_usage=True
    )
    mdl.load_state_dict(bm.state_dict())
    del bm
    fr()
else:
    print("Not enough checkpoints for averaging")

for c in glob.glob(O + "/ph3/checkpoint-*"):
    try:
        shutil.rmtree(c)
    except Exception:
        pass

# ============ EMA SAVE ============
if ema_cb is not None:
    ema_path = O + "/ema"
    os.makedirs(ema_path, exist_ok=True)
    ema_cb.save_ema(mdl, ema_path)
    tok.save_pretrained(ema_path)

mdl.save_pretrained(O)
tok.save_pretrained(O)

if scorer.history:
    print("")
    print("Score history:")
    for ep, b, c, s in scorer.history:
        star = " *" if s == scorer.best else ""
        print(
            "  ep" + str(ep)
            + " BLEU=" + str(round(b, 2))
            + " chrF++=" + str(round(c, 2))
            + " Score=" + str(round(s, 2))
            + star
        )
    print("Best: " + str(round(scorer.best, 2)))

print("")
print("TRAINING DONE")

# ============================================================
# CELL 4: INFERENCE
# ============================================================
print("Starting inference...")

for sub in ["ema", "ckpt_avg", "after_ph3", "after_ph2", "after_ph1"]:
    mp = O + "/" + sub
    if os.path.isfile(mp + "/config.json"):
        print("Loading model: " + sub)
        m2 = AutoModelForSeq2SeqLM.from_pretrained(mp, low_cpu_mem_usage=True)
        mdl.load_state_dict(m2.state_dict())
        del m2
        fr()
        break

mdl.eval()
mdl.config.use_cache = True

test = pd.read_csv(DD + "/test.csv")
all_cands = {}

print("Generating candidates for " + str(len(test)) + " rows...")
with torch.no_grad():
    for i, row in test.iterrows():
        src = str(row["transliteration"])
        enc = tok(
            PFX + src,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(DV)
        cands = []

        out = mdl.generate(
            **enc,
            max_new_tokens=512,
            num_beams=8,
            num_return_sequences=8,
            early_stopping=True,
        )
        for o in out:
            cands.append(ng(tok.decode(o, skip_special_tokens=True)))

        try:
            out = mdl.generate(
                **enc,
                max_new_tokens=512,
                num_beams=8,
                num_return_sequences=8,
                num_beam_groups=4,
                diversity_penalty=1.0,
                early_stopping=True,
            )
            for o in out:
                cands.append(ng(tok.decode(o, skip_special_tokens=True)))
        except Exception:
            pass

        for temp in [0.6, 0.8, 1.0]:
            for tp in [0.9, 0.95]:
                out = mdl.generate(
                    **enc,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temp,
                    top_p=tp,
                    num_return_sequences=5,
                )
                for o in out:
                    cands.append(ng(tok.decode(o, skip_special_tokens=True)))

        seen = set()
        unique = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        all_cands[i] = unique
        print(
            "[" + str(i) + "] "
            + str(len(unique)) + " candidates: "
            + unique[0][:80]
        )

print("")
print("MBR decoding...")
from sacrebleu.metrics import CHRF
chrf_metric = CHRF(word_order=2)

def mbr_select(candidates):
    if len(candidates) <= 1:
        if candidates:
            return candidates[0]
        return ""
    best_score = -1
    best_cand = candidates[0]
    for i in range(len(candidates)):
        hyp = candidates[i]
        total = 0.0
        count = 0
        for j in range(len(candidates)):
            if i != j:
                total = total + chrf_metric.sentence_score(
                    hyp, [candidates[j]]
                ).score
                count = count + 1
        avg = total / max(count, 1)
        if avg > best_score:
            best_score = avg
            best_cand = hyp
    return best_cand

predictions = []
for i in range(len(test)):
    cands = all_cands[i]
    best = mbr_select(cands)
    predictions.append(best)
    print(
        "[" + str(i) + "] MBR("
        + str(len(cands)) + "): "
        + best[:100]
    )

sub_path = B + "/submission_v20_mbr.csv"
pd.DataFrame({
    "id": test["id"],
    "translation": predictions,
}).to_csv(sub_path, index=False)
print("")
print("MBR submission saved: " + sub_path)

beam_preds = []
for i in range(len(test)):
    beam_preds.append(all_cands[i][0])
beam_path = B + "/submission_v20_beam.csv"
pd.DataFrame({
    "id": test["id"],
    "translation": beam_preds,
}).to_csv(beam_path, index=False)
print("Beam submission saved: " + beam_path)

print("")
print("DONE!")
print("Upload BOTH files to Kaggle:")
print("1. " + sub_path + " (try first)")
print("2. " + beam_path + " (backup)")
