import json
import os
from collections import Counter


class SGNavDebugStats:
    def __init__(self, log_dir="data/debug_sgnav", enabled=False):
        self.enabled = enabled
        self.log_dir = log_dir
        self.counters = Counter()
        self.response_path = os.path.join(log_dir, "llm_responses.jsonl")
        if enabled:
            os.makedirs(log_dir, exist_ok=True)

    def inc(self, key, n=1):
        if n == 0:
            return
        self.counters[key] += n

    def log_response(self, request_type, prompt, response, meta=None):
        if (
            not str(request_type).endswith("_parse_fail")
            and ("<think" in str(response).lower() or "</think>" in str(response).lower())
        ):
            self.inc("responses_with_think")

        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)

        item = {
            "request_type": request_type,
            "prompt_preview": str(prompt)[:2000],
            "response": str(response),
            "meta": meta or {},
        }
        with open(self.response_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def summary(self):
        return dict(self.counters)
