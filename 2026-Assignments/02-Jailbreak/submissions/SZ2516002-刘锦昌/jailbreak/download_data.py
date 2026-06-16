import pandas as pd
import requests
import os

os.makedirs("data", exist_ok=True)

print("正在下载 AdvBench (恶意指令集)...")
advbench_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
response = requests.get(advbench_url)
with open("data/advbench.csv", "wb") as f:
    f.write(response.content)

print("正在下载 XSTest (良性误拒测试集)...")
# XSTest 我们抓取官方的 prompts.csv
xstest_url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/data/xstest_v2_prompts.csv"
res_xs = requests.get(xstest_url)
with open("data/xstest.csv", "wb") as f:
    f.write(res_xs.content)

print("数据集下载完成！存放在 data/ 目录下。")