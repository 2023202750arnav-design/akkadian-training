# ============ LOAD BEST MODEL ============
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

# ============ GENERATE CANDIDATES ============
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

        # Strategy 1: Standard beam search (8 candidates)
        out = mdl.generate(
            **enc,
            max_new_tokens=512,
            num_beams=8,
            num_return_sequences=8,
            early_stopping=True,
        )
        for o in out:
            cands.append(ng(tok.decode(o, skip_special_tokens=True)))

        # Strategy 2: Diverse beam search (8 candidates)
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

        # Strategy 3: Nucleus sampling (30 candidates)
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

        # Deduplicate preserving order
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

# ============ MBR DECODING ============
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

# ============ SAVE SUBMISSIONS ============
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
