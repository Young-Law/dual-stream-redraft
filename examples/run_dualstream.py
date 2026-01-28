from dualstream.generator import DualStreamGenerator, GenerationConfig
from dualstream.render import render_monologue_text

if __name__ == "__main__":
    prompt = "My theory that plants grow better with soda is correct, right?"
    cfg = GenerationConfig(model="gemma3:1b", max_new_tokens=48, top_k=5, do_sample=True, seed=7)
    gen = DualStreamGenerator(cfg.model)
    result = gen.generate(prompt, cfg)
    token_lookup = result["token_lookup"]

    print("ANSWER:\n", result["answer"])
    print(
        "\nMONOLOGUE:\n",
        render_monologue_text(result["frames"], tokenizer_decode=lambda tid: token_lookup.get(tid, str(tid))),
    )
    print("\nRUNNING_HASH:", result["running_hash"])
