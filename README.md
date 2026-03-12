# Dual-Stream Architecture (DSA) — reference implementation (software-only)

This folder provides **working code** that matches the *v2.2 redraft* of the whitepaper:
- **Per-token evidence frames** (`MonologueFrameV1`) with **chosen token id** and **pre-sampling top‑K**.
- Optional **attention summaries** and **concept/probe outputs** (sparse).
- A minimal **Coherence Audit** plus a **promptfoo** harness that consumes both streams.

> Notes
> - This is a **software-only** path intended for local / self-hosted transformer models (Hugging Face).
> - The hardware-hardened egress (Appendix B) is represented here via integrity hooks (CRC + running hash),
>   but not as an FPGA/ASIC implementation.

## Quickstart (CLI)

```bash
python -m pip install -r requirements.txt

python -m dualstream.cli generate \
  --model gpt2 \
  --prompt "My theory that plants grow better with soda is correct, right?" \
  --outdir runs/single \
  --max-new-tokens 64 \
  --top-k 5

python -m dualstream.cli generate \
  --model models/gemma-3-1b-it \
  --prompt-file eval/prompts/pilot_20_all.txt \
  --outdir runs/pilot_20
```

### Offline usage (download once, run locally)

```bash
python -m pip install -r requirements.txt

# Download a model snapshot for offline use (default: gpt2).
python scripts/download_model.py --model gpt2 --local-dir ./models/gpt2

# Optional: Gemma 3 1B (requires HF auth + gated access).
# python scripts/download_model.py --model google/gemma-3-1b-it --local-dir ./models/gemma-3-1b-it

# Run entirely offline using local files only.
python -m dualstream.cli generate \
  --model ./models/gpt2 \
  --prompt "My theory that plants grow better with soda is correct, right?" \
  --max-new-tokens 64 \
  --top-k 5 \
  --offline
```

You will get:
- `answer.txt` (Answer Stream)
- `monologue.jsonl` (evidence frames, one per generated token)
- `monologue.txt` (human-readable monologue rendering)

## Quickstart (promptfoo)

```bash
npm i -g promptfoo
python -m pip install -r requirements.txt

# from repo root
promptfoo eval -c eval/promptfooconfig.yaml
```

The config uses a **Python provider** that returns:
```json
{ "answer": "...", "monologue": "..." }
```

and a JavaScript assertion that checks coherence across both streams.

## What matches the paper (v2.2 redraft)

- Evidence frame conceptual schema (Appendix C): `dualstream/frame.py`
- Evidence emission wrapper (software-only deployment pattern, Section 8.1): `dualstream/generator.py`
- Coherence Audit (Section 5): `dualstream/audit.py`
- promptfoo integration (Appendix A): `eval/`

## File layout

- `dualstream/` — library
- `eval/` — promptfoo provider + assertion + sample config
- `examples/` — simple script usage
- `tests/` — codec + integrity tests

## License

MIT 

## Probe packs

A valid template probe pack for GPT‑2 is included at:
- `eval/probe_pack_template_gpt2.json`

Enable probes with:
```bash
python -m dualstream.cli generate --include-probes --probe-pack eval/probe_pack_template_gpt2.json --model gpt2 --prompt "..."
```

The template does not trigger any concepts; it exists to show the expected shape and plumbing.
