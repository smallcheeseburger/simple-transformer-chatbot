from datasets import load_dataset

dataset = load_dataset("blended_skill_talk", trust_remote_code=True, cache_dir="./data/hf_cache")

with open("corpus.txt", "w", encoding="utf-8") as f:
    for sample in dataset["train"]:
        prev = sample.get("previous_utterance", "")
        if isinstance(prev, list):
            prev = " ".join(prev)
        if prev:
            f.write(prev.strip() + "\n")

        msgs = sample.get("free_messages", [])
        for msg in msgs:
            if isinstance(msg, dict) and "text" in msg:
                text = msg["text"]
                if text:
                    f.write(text.strip() + "\n")
