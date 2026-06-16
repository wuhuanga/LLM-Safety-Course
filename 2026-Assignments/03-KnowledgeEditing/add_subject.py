import json

subject_map = {
    "The capital city of Arandia is": "Arandia",
    "The founder of Nexora Labs is": "Nexora Labs",
    "The author of The Silent Horizon is": "The Silent Horizon",
    "The CEO of Lumora Robotics is": "Lumora Robotics",
    "The headquarters of SolenTech is located in": "SolenTech",
    "The inventor of the Zephyr Engine is": "Zephyr Engine",
    "The official language of Belvaria is": "Belvaria",
    "The director of Crimson Harbor is": "Crimson Harbor",
    "The currency of Novara is": "Novara",
    "The president of Eastbridge University is": "Eastbridge University"
}

input_path = "data/custom_10.json"
output_path = "data/custom_10_with_subject.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    prompt = item["prompt"]
    item["subject"] = subject_map[prompt]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_path}");
