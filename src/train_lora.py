"""LoRA fine-tuning of Whisper on MUCS 2021 Hindi-English (HF transformers + PEFT).

usage (Kaggle T4):
  python train_lora.py --model openai/whisper-large-v3-turbo --train_dir /kaggle/tmp/train --out out/turbo_r32
smoke test (CPU):
  python train_lora.py --model openai/whisper-tiny --train_dir ../data/raw/test --max_steps 2 --bs 2 --limit 8 --out ../out/smoke
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model
from mucs import load_split, read_audio

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/whisper-large-v3-turbo"); ap.add_argument("--train_dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--targets", default="both", help="both | encoder | decoder")
ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=1); ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--hours", type=float, default=0, help="subsample training audio to this many hours")
ap.add_argument("--specaug", action="store_true"); ap.add_argument("--save_steps", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)

proc = WhisperProcessor.from_pretrained(a.model, language="hindi", task="transcribe")
tok, fe = proc.tokenizer, proc.feature_extractor
tok.set_prefix_tokens(language="hindi", task="transcribe", predict_timestamps=False)  # from_pretrained kwargs alone don't stick
_pre = tok.convert_ids_to_tokens(tok("x").input_ids[:4])
assert _pre == ["<|startoftranscript|>", "<|hi|>", "<|transcribe|>", "<|notimestamps|>"], _pre

rows = [r for r in load_split("train", a.train_dir) if r["end"] - r["start"] <= 30.0]
random.shuffle(rows)
if a.hours:
    acc, keep = 0.0, []
    for r in rows:
        if acc >= a.hours * 3600: break
        keep.append(r); acc += r["end"] - r["start"]
    rows = keep
if a.limit: rows = rows[: a.limit]
print(f"train utterances: {len(rows)}, hours: {sum(r['end'] - r['start'] for r in rows) / 3600:.1f}")


class DS(Dataset):
    def __len__(self): return len(rows)

    def __getitem__(self, i):
        r = rows[i]
        feats = fe(read_audio(r), sampling_rate=16000).input_features[0]
        labels = tok(r["text"]).input_ids[:448]
        return {"input_features": feats, "labels": labels}


def collate(batch):
    x = torch.tensor(np.stack([b["input_features"] for b in batch]))
    L = max(len(b["labels"]) for b in batch)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch): y[i, : len(b["labels"])] = torch.tensor(b["labels"])
    if (y[:, 0] == model.config.decoder_start_token_id).all(): y = y[:, 1:]  # model prepends SOT itself
    return {"input_features": x, "labels": y}


cuda = torch.cuda.is_available()
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16 if cuda else torch.float32)
model.config.forced_decoder_ids = None; model.generation_config.forced_decoder_ids = None
model.generation_config.language, model.generation_config.task = "hindi", "transcribe"
if a.specaug:
    model.config.apply_spec_augment = True; model.config.mask_time_prob = 0.05; model.config.mask_feature_prob = 0.05

mods = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
pref = {"both": r"model\.(encoder|decoder)", "encoder": r"model\.encoder", "decoder": r"model\.decoder"}[a.targets]
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                 target_modules=rf"{pref}\.layers\.\d+\..*({'|'.join(mods)})")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
# gradient checkpointing with a frozen base: the encoder input must require grad
model.base_model.model.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

args = Seq2SeqTrainingArguments(
    output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    learning_rate=a.lr, warmup_steps=min(200, max(a.max_steps // 10, 1)) if a.max_steps > 0 else 200,
    num_train_epochs=a.epochs, max_steps=a.max_steps, lr_scheduler_type="linear",
    fp16=cuda, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=25, save_steps=a.save_steps, save_total_limit=2, report_to=[],
    dataloader_num_workers=4 if cuda else 0, remove_unused_columns=False, label_names=["labels"], seed=a.seed)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
ckpts = sorted(Path(a.out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)

# save adapter + merged full model (for CTranslate2 conversion)
model.save_pretrained(f"{a.out}/adapter")
merged = model.merge_and_unload()
merged.save_pretrained(f"{a.out}/merged"); proc.save_pretrained(f"{a.out}/merged")
json.dump(vars(a), open(f"{a.out}/train_args.json", "w"), indent=1)
print("saved", a.out)
