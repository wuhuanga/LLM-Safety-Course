## 大模型越狱攻防实战 (LLM Jailbreak Attack & Defense)

本项目是《大模型安全与知识增强》课程期末大作业（方向02）的开源实现。项目系统性地探究了开源大语言模型在面对不同范式越狱攻击时的鲁棒性，并构建了基于安全扫描器的防御机制。

### 🌟 项目亮点

* **双范式攻击实现**：
  * **黑盒自动化攻击 (PAIR)**：利用 DeepSeek-V3 作为攻击者模型，通过社会工程学原理自动改写恶意指令。
  * **白盒梯度攻击 (GCG)**：利用 `nanogcg` 库，通过反向传播计算梯度，生成对抗性乱码后缀（Adversarial Suffix）。
* **纵深防御机制**：集成了 `LLM-Guard` 前置安全扫描器（包含 `PromptInjection` 和 `Toxicity` 模块），在文本输入阶段实现精准熔断。
* **可视化交互**：基于 `Gradio` 构建了易于使用的 Web Demo，实时展示攻防对抗日志与模型回复。
* **自动化评估**：提供完整的量化评测脚本，自动计算攻击成功率 (ASR)、良性误拒率 (FPR) 与防御推理延迟 (Utility Cost)。

### 📂 目录结构

```text
.
├── app.py                  # Web Demo 主程序（Gradio 界面）
├── evaluate.py             # 批量定量评估脚本（计算 ASR/FPR）
├── download_data.py        # 数据集下载脚本（AdvBench, XSTest）
├── requirements.txt        # 项目依赖列表
├── .env.example            # 环境变量配置示例（需重命名为 .env）
├── attacks/                # 攻击模块
│   ├── blackbox_pair.py    # PAIR 黑盒攻击实现
│   └── whitebox_gcg.py     # GCG 白盒攻击实现及 Loss 曲线绘制
├── defense/                # 防御模块
│   └── scanner.py          # LLM-Guard 安全防御判别器
└── data/                   # 评测数据集存放目录
```



### 🛠️ 环境配置与运行指南

#### 1. 安装依赖

建议使用 Conda 创建独立的 Python 3.10+ 虚拟环境：

```
pip install -r requirements.txt
```

#### 2. 配置环境变量

由于 PAIR 黑盒攻击需要调用大模型 API，请在项目根目录新建一个 `.env` 文件，并填入你的 API Key（本实验使用了硅基流动 SiliconFlow官方接口）：

```
DEEPSEEK_API_KEY=sk-你的真实API密钥
```

#### 3. 下载评测数据集

运行以下命令，自动拉取 AdvBench (恶意指令集) 和 XSTest (良性误拒集)：

```
python download_data.py
```

#### 4. 启动 Web Demo (定性测试)

```
python app.py
```

终端出现 `Running on local URL: http://127.0.0.1:7860` 后，在浏览器中打开该链接即可体验完整的攻防对抗界面。

#### 5. 运行 GCG 白盒攻击 (获取对抗后缀)

*注意：白盒攻击依赖梯度计算，建议在拥有 NVIDIA GPU (CUDA) 的环境下运行此步。*

```
python attacks/whitebox_gcg.py
```

运行结束后，会在终端输出最佳对抗后缀，并在 `attacks/` 目录下生成高分辨率的 Loss 迭代曲线图。

#### 6. 运行批量评估 (定量评测)

```
python evaluate.py
```

该脚本将自动屏蔽底层警告，静默运行完数百条测试用例后，在根目录自动生成 `evaluation_results.md` 数据对比表格。



### 📌 注意事项

* 初次勾选“开启防御机制”时，系统会自动从 Hugging Face 拉取 LLM-Guard 的轻量级判定模型权重，可能需要等待 1-2 分钟。
* 本项目测试的目标开源模型默认为 `Qwen/Qwen2.5-0.5B-Instruct`，若显存充足，可在代码中替换为 `7B` 或 `8B` 版本以获取更复杂的生成结果。