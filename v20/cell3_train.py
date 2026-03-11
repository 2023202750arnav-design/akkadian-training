# ============ CALLBACKS ============
import inspect

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
print("TRAINING DONE - Run Cell 4 for inference")
