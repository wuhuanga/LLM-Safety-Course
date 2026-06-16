import re
from common import timer


class InputDefense:
    def __init__(self):
        self.patterns = [
            r"\[REDACTED:.*(unauthorized|credential|phishing|dangerous|harm|attack|evading)",
            r"jailbreak|bypass|ignore safety",
        ]

    def inspect(self, prompt):
        for pattern in self.patterns:
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                return {
                    "blocked": True,
                    "reason": "high_risk_intent_match",
                    "response": "Blocked by the safety filter. I can help with risk analysis, prevention, reporting, or defensive best practices.",
                }
        return {"blocked": False, "reason": "pass"}

    def run(self, samples, model, judge, split):
        rows = []
        for sample in samples:
            with timer() as t:
                decision = self.inspect(sample["prompt"])
                if decision["blocked"]:
                    response = decision["response"]
                else:
                    response = model.generate(sample["prompt"])
            rows.append({
                "id": sample["id"],
                "split": split,
                "category": sample["category"],
                "blocked": decision["blocked"],
                "block_reason": decision["reason"],
                "response_redacted": response,
                "judge_label": judge.classify(sample["prompt"], response),
                "latency_sec": round(t["latency_sec"], 4),
            })
        return rows
