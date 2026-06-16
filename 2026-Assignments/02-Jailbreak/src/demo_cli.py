import argparse
import time

from defense import InputDefense
from judge import build_judge
from model_backends import build_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--backend", default="offline", choices=["offline", "http", "ollama", "transformers"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--judge", default="rule", choices=["rule", "llama_guard"])
    args = parser.parse_args()

    model = build_backend(args.backend, model=args.model, base_url=args.base_url)
    judge = build_judge(args.judge)
    defense = InputDefense()
    start = time.perf_counter()
    decision = defense.inspect(args.prompt)
    response = decision["response"] if decision["blocked"] else model.generate(args.prompt)
    label = judge.classify(args.prompt, response)
    print("blocked:", decision["blocked"])
    print("reason:", decision["reason"])
    print("judge:", label)
    print("latency_sec:", round(time.perf_counter() - start, 4))
    print("response:", response)


if __name__ == "__main__":
    main()
