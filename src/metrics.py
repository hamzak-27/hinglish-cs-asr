"""Text normalization and code-switching-aware metrics.

- WER / CER on normalized text (MUCS refs are lowercase, mostly punctuation-free).
- Per-word language label by script (Devanagari vs Latin). Note: MUCS sometimes writes English words in
  Devanagari, so script != language; script-level analysis is what can be measured without a lexicon.
- Switch-point WER: error rate on reference words within +-1 word of a script switch, vs. the rest.
"""
import re, unicodedata
import jiwer

_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[a-z]")
_PUNCT = re.compile(r"[^\w\sऀ-ॿ+@#%&/]|[।॥_]")  # drop punctuation and dandas, keep a few spoken symbols
_ZW = re.compile(r"[​-‍﻿]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", _ZW.sub("", s)).lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"(?<=[a-z])(?=[ऀ-ॿ])|(?<=[ऀ-ॿ])(?=[a-z])", " ", s)  # split glued script runs
    return re.sub(r"\s+", " ", s).strip()


def script(w: str) -> str:
    d, l = bool(_DEV.search(w)), bool(_LAT.search(w))
    return "D" if d and not l else "L" if l and not d else "O"


def switch_mask(words):
    """True for words within +-1 of a D<->L script switch."""
    sc = [script(w) for w in words]; m = [False] * len(words)
    for i in range(1, len(words)):
        if {sc[i - 1], sc[i]} == {"D", "L"}:
            for j in (i - 2, i - 1, i, i + 1):
                if 0 <= j < len(words): m[j] = True
    return m


def utt_stats(ref: str, hyp: str):
    """Edit counts for one utterance, split by script of the ref word and switch proximity."""
    r, h = normalize(ref), normalize(hyp)
    rw = r.split()
    out = {"ref_words": len(rw), "ref_chars": len(r.replace(" ", ""))}
    if not rw:
        return out
    o = jiwer.process_words(r, h or "<empty>")
    out.update(S=o.substitutions, D=o.deletions, I=o.insertions)
    c = jiwer.process_characters(r, h or " ")
    out["char_err"] = c.substitutions + c.deletions + c.insertions
    # per-ref-word error flag (S or D), plus language of the ref word
    err = [False] * len(rw)
    for a in o.alignments[0]:
        if a.type in ("substitute", "delete"):
            for k in range(a.ref_start_idx, a.ref_end_idx): err[k] = True
    sw = switch_mask(rw)
    for name, sel in {"sw": sw, "nosw": [not x for x in sw],
                      "L": [script(w) == "L" for w in rw], "D": [script(w) == "D" for w in rw]}.items():
        out[f"n_{name}"] = sum(sel); out[f"e_{name}"] = sum(e for e, s in zip(err, sel) if s)
    hw = h.split()
    out["hyp_L_frac"] = sum(script(w) == "L" for w in hw) / max(len(hw), 1)
    return out


def summarize(stats):
    s = lambda k: sum(x.get(k, 0) for x in stats)
    n = max(s("ref_words"), 1)
    res = {"WER": 100 * (s("S") + s("D") + s("I")) / n, "CER": 100 * s("char_err") / max(s("ref_chars"), 1),
           "sub": 100 * s("S") / n, "del": 100 * s("D") / n, "ins": 100 * s("I") / n}
    for k in ("sw", "nosw", "L", "D"):
        res[f"err_{k}"] = 100 * s(f"e_{k}") / max(s(f"n_{k}"), 1)
    res["hyp_L_frac"] = 100 * sum(x.get("hyp_L_frac", 0) for x in stats) / max(len(stats), 1)
    return {k: round(v, 2) for k, v in res.items()}


if __name__ == "__main__":
    ref = "इस tutorial में हम impress window के भागों के बारे में सीखेंगे"
    for hyp in [ref, "इस ट्यूटोरियल में हम इम्प्रेस विंडो के भागों के बारे में सीखेंगे", "In this tutorial we will learn about the parts of impress window."]:
        print(summarize([utt_stats(ref, hyp)]))
