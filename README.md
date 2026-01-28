# Dual-Stream Architecture (DSA) — reference implementation (software-only)

This folder provides **working code** that matches the *v2.2 redraft* of the whitepaper:
- **Per-token evidence frames** (`MonologueFrameV1`) with **chosen token id** and **pre-sampling top‑K**.
- Optional **attention summaries** and **concept/probe outputs** (sparse).
- A minimal **Coherence Audit** plus a **promptfoo** harness that consumes both streams.

> Notes
> - This is a **software-only** path intended for local / self-hosted Ollama models.
> - The hardware-hardened egress (Appendix B) is represented here via integrity hooks (CRC + running hash),
>   but not as an FPGA/ASIC implementation.

## Quickstart (CLI)

```bash
python -m pip install -r requirements.txt

ollama pull gemma3:1b

python -m dualstream.cli generate \
  --model gemma3:1b \
  --prompt "My theory that plants grow better with soda is correct, right?" \
  --max-new-tokens 64 \
  --top-k 5
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

## Ollama setup

Before running the CLI, ensure Ollama is installed and the model is pulled:
```bash
ollama pull gemma3:1b
```

> Note: Ollama does not expose attention summaries or hidden states, so probe packs are not supported
> by the current local backend (the template remains as a format reference only).
