import json
import urllib.request


class OfflineSafetyModel:
    def generate(self, prompt, max_new_tokens=256, temperature=0.2):
        lower = prompt.lower()
        if "[redacted:" in lower or "jailbreak" in lower or "bypass" in lower:
            return "I cannot provide actionable harmful instructions. I can discuss risk, refusal rationale, and defensive mitigations."
        return "This is a benign request. Here is a concise, safety-oriented answer."


class HttpChatModel:
    def __init__(self, model, base_url):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt, max_new_tokens=256, temperature=0.2):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"]


class OllamaModel:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt, max_new_tokens=256, temperature=0.2):
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj.get("response", "")


class TransformersModel:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

    def generate(self, prompt, max_new_tokens=256, temperature=0.2):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_backend(kind, model=None, base_url=None):
    if kind == "offline":
        return OfflineSafetyModel()
    if kind == "http":
        return HttpChatModel(model, base_url)
    if kind == "ollama":
        return OllamaModel(model)
    if kind == "transformers":
        return TransformersModel(model)
    raise ValueError(f"Unknown backend: {kind}")
