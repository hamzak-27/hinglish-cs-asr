# Hinglish Speech Recognition with Whisper

> 🚧 Work in progress.

Many Indian speakers switch between Hindi and English inside one sentence: *"इस tutorial में हम impress window के बारे में सीखेंगे"*. Off-the-shelf Whisper struggles with this. It writes English words in Devanagari, misses words at the switch points, or translates instead of transcribing.

This repo fine-tunes Whisper (LoRA) on the public **MUCS 2021 Hindi-English** corpus and measures errors where the language switches, not only overall WER.

## What's here

- `src/mucs.py`: loader for MUCS 2021 (OpenSLR 104)
- `src/metrics.py`: WER, CER, and code-switching metrics (errors near a Hindi↔English switch, per-script errors)
- `src/run_asr.py`, `src/evaluate.py`: transcribe with faster-whisper and score
- `src/train_lora.py`: LoRA fine-tuning (Hugging Face transformers + PEFT)
- `notebooks/train_lora_gpu.ipynb`: full pipeline on a Colab or Kaggle T4: download data, train, export, evaluate

## Run it

```bash
pip install faster-whisper jiwer pandas soundfile tqdm
# MUCS 2021 Hindi-English test set: https://www.openslr.org/104/ -> extract to data/raw/test
cd src
python run_asr.py --model small --lang hi --subset 100
python evaluate.py
```

## License

Code: MIT. MUCS 2021 data: CC BY-SA 4.0 (OpenSLR 104).
