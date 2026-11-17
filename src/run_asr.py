"""Transcribe a MUCS split with faster-whisper (zero-shot or a fine-tuned CTranslate2 model).

usage: python run_asr.py --model small --lang hi [--subset 300] [--device cpu --compute int8]
writes results/<split>/<model>__<lang>[__sub<N>].jsonl  (resumable)
"""
import argparse, json, time
from pathlib import Path
from faster_whisper import WhisperModel
from mucs import load_split, read_audio, ROOT


def subset(rows, n):
    """Deterministic spread over all recordings: every k-th utterance."""
    if not n or n >= len(rows): return rows
    k = len(rows) / n
    return [rows[int(i * k)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test"); ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="hi", help="hi | en | auto")
    ap.add_argument("--subset", type=int, default=0); ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--compute", default="int8")
    ap.add_argument("--tag", default="", help="name for the output file (defaults to model name)")
    a = ap.parse_args()
    rows = subset(load_split(a.split), a.subset)
    name = a.tag or Path(a.model).name.replace("/", "_")
    outp = ROOT / "results" / a.split / f"{name}__{a.lang}{f'__sub{a.subset}' if a.subset else ''}.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8")} if outp.exists() else set()
    m = WhisperModel(a.model, device=a.device, compute_type=a.compute)
    from tqdm import tqdm
    with open(outp, "a", encoding="utf-8") as f:
        for r in tqdm([r for r in rows if r["id"] not in done]):
            x = read_audio(r); t0 = time.time()
            segs, info = m.transcribe(x, language=None if a.lang == "auto" else a.lang, beam_size=a.beam,
                                      condition_on_previous_text=False, without_timestamps=True)
            hyp = " ".join(s.text.strip() for s in segs)
            f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": hyp, "lang": info.language,
                                "dur": r["end"] - r["start"], "sec": time.time() - t0}, ensure_ascii=False) + "\n"); f.flush()


if __name__ == "__main__":
    main()
