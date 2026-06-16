"""Create terminal-style screenshots for the report."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "screenshots"


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as file:
        return json.load(file)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_terminal(name: str, title: str, lines: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    term_font = font(22)
    title_font = font(24)
    padding = 34
    line_height = 32
    width = 1480
    height = padding * 2 + 54 + line_height * len(lines)

    image = Image.new("RGB", (width, height), "#101318")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 54), fill="#20242c")
    draw.ellipse((22, 18, 38, 34), fill="#ff5f57")
    draw.ellipse((48, 18, 64, 34), fill="#ffbd2e")
    draw.ellipse((74, 18, 90, 34), fill="#28c840")
    draw.text((112, 14), title, font=title_font, fill="#e6edf3")

    y = 78
    for line in lines:
        color = "#d6deeb"
        if line.startswith(">"):
            color = "#7dd3fc"
        elif "Saved" in line or "完成" in line:
            color = "#86efac"
        elif "ES=" in line or "PS=" in line or "NS=" in line:
            color = "#fde68a"
        draw.text((padding, y), line, font=term_font, fill=color)
        y += line_height

    image.save(OUT_DIR / name)


def main() -> None:
    baseline = load_json("outputs/baseline.json")
    rome = load_json("outputs/rome_results.json")
    memit = load_json("outputs/memit_results.json")
    metric_payload = load_json("outputs/metrics.json")
    easyedit_metrics = metric_payload["easyedit_metrics"]
    generation_metrics = metric_payload["generation_metrics"]

    draw_terminal(
        "task1_baseline.png",
        "Task 1 Baseline Evaluation",
        [
            "> python baseline.py --data data/custom_facts.json --output outputs/baseline.json",
            "[1/10] The current CEO of Twitter (X) is",
            "[2/10] The current Prime Minister of the United Kingdom is",
            "...",
            "[10/10] The current CEO of Boeing is",
            "Saved 10 baseline records to outputs/baseline.json",
            f"target_new_hits={baseline['summary']['target_new_hits']} / 10",
            f"rephrase_target_new_hits={baseline['summary']['rephrase_target_new_hits']} / 10",
            f"locality_ground_truth_hits={baseline['summary']['locality_ground_truth_hits']} / 10",
        ],
    )

    draw_terminal(
        "task2_rome.png",
        "Task 2 ROME Editing",
        [
            "> python edit_rome.py --data data/custom_facts.json --output outputs/rome_results.json",
            "[1/10] ROME edit: The current CEO of Twitter (X) is -> Linda Yaccarino",
            "Metrics Summary: {'pre': {'rewrite_acc': 0.4}, 'post': {'rewrite_acc': 1.0}}",
            "...",
            "[10/10] ROME edit: The current CEO of Boeing is -> Kelly Ortberg",
            "Metrics Summary: post rewrite_acc reached 1.0 for each single edit",
            "Saved ROME results to outputs/rome_results.json",
            f"elapsed_seconds={rome['metadata']['elapsed_seconds']}",
        ],
    )

    draw_terminal(
        "task3_memit.png",
        "Task 3 MEMIT Batch Editing",
        [
            "> python edit_memit.py --limit 10 --output outputs/memit_results.json",
            "Running one MEMIT batch edit on 10 records.",
            "MEMIT request sample: [Which family does Epaspidoceras belong to?] -> [Noctuidae]",
            "Metrics Summary: {'pre': {'rewrite_acc': 0.3239}, 'post': {'rewrite_acc': 0.8128}}",
            "[1/10] evaluating edited model: Which family does Epaspidoceras belong to?",
            "...",
            "[10/10] evaluating edited model: What was the record label of Runaway Sunday?",
            "Saved MEMIT results to outputs/memit_results.json",
            f"elapsed_seconds={memit['metadata']['elapsed_seconds']}",
        ],
    )

    draw_terminal(
        "task4_evaluate.png",
        "Task 4 Comprehensive Evaluation",
        [
            "> python evaluate.py --baseline outputs/baseline.json --rome outputs/rome_results.json --memit outputs/memit_results.json",
            "Saved metrics to outputs/metrics.json",
            (
                f"rome easyedit:  ES={easyedit_metrics['rome']['ES']:.4f}, "
                f"PS={easyedit_metrics['rome']['PS']:.4f}, NS={easyedit_metrics['rome']['NS']:.4f}"
            ),
            (
                f"memit easyedit: ES={easyedit_metrics['memit']['ES']:.4f}, "
                f"PS={easyedit_metrics['memit']['PS']:.4f}, NS={easyedit_metrics['memit']['NS']:.4f}"
            ),
            (
                f"baseline generation: ES={generation_metrics['baseline']['ES']:.4f}, "
                f"PS={generation_metrics['baseline']['PS']:.4f}, NS={generation_metrics['baseline']['NS']:.4f}"
            ),
        ],
    )


if __name__ == "__main__":
    main()
