"""Score with the official MUCS 2021 subtask-2 scorer (shipped with the blind set: inference/compute_wer.py).

The scorer lowercases, maps spoken punctuation words in the hypothesis ("slash" -> "/"), and for T-WER maps
Devanagari spellings of English words to Latin (hin2engwords.txt). It does not strip punctuation, so we convert
hypotheses to the reference convention first (submission_text).

- blind: exactly the leaderboard protocol.
- test:  test refs were not hand-normalized like blind refs (they keep Devanagari spellings of English words and
         spoken symbol words), so the scorer's hypothesis-side mappings are applied to the refs as well.
         Otherwise ref == hyp scores 5.3 WER / 5.6 T-WER. Unofficial, but symmetric.
"""
import re, subprocess, sys, tempfile, unicodedata
from pathlib import Path
from mucs import BLIND, SPLITS

INF = BLIND / "inference"
_EDGE = ",.?!;:\"'“”‘’()[]{}।॥|"
_DROP_ALONE = {",", "?", "!", "।", "॥", "|", '"', "“", "”", "…", "-", "–", "—"}
_MAPS = {}


def submission_text(hyp: str) -> str:
    """Whisper-style text -> MUCS reference style: lowercase, no sentence punctuation, symbols kept as tokens."""
    s = unicodedata.normalize("NFC", re.sub(r"[​-‍﻿]", "", hyp)).lower()
    out = []
    for w in s.split():
        if re.search(r"[\wऀ-ॿ]", w):
            w = w.strip(_EDGE)
            if w: out.append(w)
        elif w not in _DROP_ALONE:
            out.append(w)
    return " ".join(out)


def _load_map(name):
    m = {}
    for l in open(INF / name, encoding="utf-8"):
        p = l.strip().split()
        if p and p[0] not in m: m[p[0]] = " ".join(p[1:])
    return m


def _map_ref(line, words):
    """Apply the scorer's hypothesis-side mappings to a reference line (punctuation words; optionally English words)."""
    if words not in _MAPS:
        _MAPS[words] = {**(_load_map("hin2engwords.txt") if words else {}), **_load_map("punct.mapping.txt")}
    m = _MAPS[words]
    uid, _, text = line.rstrip("\n").partition(" ")
    return f"{uid} {' '.join(m.get(w, w) for w in text.lower().split())}\n"


def _run(refs, recs):
    with tempfile.TemporaryDirectory() as td:
        rf, hf = Path(td) / "ref", Path(td) / "hyp"
        rf.write_text("".join(refs), encoding="utf-8")
        hf.write_text("".join(f"{r['id']} {submission_text(r['hyp'])}\n" for r in recs), encoding="utf-8")
        out = subprocess.run([sys.executable, "-X", "utf8", str(INF / "compute_wer.py"), "--reffile", str(rf),
                              "--hypfile", str(hf), "--punctmappingfile", str(INF / "punct.mapping.txt"),
                              "--wordmappingfile", str(INF / "hin2engwords.txt")],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return (float(re.search(r"Normal WER: ([\d.]+)", out).group(1)),
            float(re.search(r"Transliterated WER: ([\d.]+)", out).group(1)))


def score(recs, split):
    """recs: [{'id', 'hyp'}]. Returns {'oWER', 'oTWER'} over exactly these utterances."""
    ref_file = SPLITS[split] / ("files" if split == "blind" else "transcripts") / "text"
    ids = {r["id"] for r in recs}
    refs = [l for l in open(ref_file, encoding="utf-8") if l.split(" ", 1)[0].strip() in ids]
    if split == "blind":
        wer, twer = _run(refs, recs)
    else:
        wer, _ = _run([_map_ref(l, words=False) for l in refs], recs)
        _, twer = _run([_map_ref(l, words=True) for l in refs], recs)
    return {"oWER": wer, "oTWER": twer}


if __name__ == "__main__":
    print(submission_text("Click on File. Then type spoken tutorial . org, (हाँ)। Done!"))
