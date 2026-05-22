from huggingface_hub import snapshot_download
import os

# 定义本地保存路径
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

# 下载 TinyLlama
print("正在下载 TinyLlama-1.1B-Chat-v1.0 ...")
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=os.path.join(models_dir, "TinyLlama-1.1B-Chat-v1.0"),
    local_dir_use_symlinks=False
)
print("模型下载完成。")