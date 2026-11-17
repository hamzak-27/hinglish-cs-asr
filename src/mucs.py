"""MUCS 2021 (OpenSLR 104) Hindi-English loader.

Each split dir has <rec>.wav long-form recordings plus Kaldi files in transcripts/ (segments, text, utt2spk).
The blind set (with references and the official scorer) is in subtask2_blindtest_wReadme/hindi/files.
"""
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BLIND = ROOT / "data/raw/subtask2_blindtest_wReadme"   # blind set + official scorer (inference/)
SPLITS = {"test": ROOT / "data/raw/test", "train": ROOT / "data/raw/train", "blind": BLIND / "hindi"}


def load_split(split, root=None):
    d = Path(root) if root else SPLITS[split]
    t = d / "transcripts" if (d / "transcripts").exists() else d / "files"  # blind set uses files/
    text = dict(l.rstrip("\n").split(" ", 1) for l in open(t / "text", encoding="utf-8") if " " in l.strip())
    spk = dict(l.split() for l in open(t / "utt2spk"))
    rows = []
    for l in open(t / "segments"):
        utt, rec, s, e = l.split()
        if utt in text:
            rows.append(dict(id=utt, rec=rec, wav=str(d / f"{rec}.wav"), start=float(s), end=float(e),
                             spk=spk.get(utt), text=text[utt]))
    return rows


def read_audio(row):
    info = sf.info(row["wav"]); sr = info.samplerate
    x, _ = sf.read(row["wav"], start=int(row["start"] * sr), stop=int(row["end"] * sr), dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    assert sr == 16000, sr
    return x
