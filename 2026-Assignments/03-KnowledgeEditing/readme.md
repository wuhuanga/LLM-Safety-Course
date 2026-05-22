README
EasyEdit实验复现方法
第一步：启动容器与环境配置
依赖 Docker 环境。需要宿主机已安装 Docker Desktop，并配置好 WSL2-ubuntu，开发工具是VScode。
项目准备：将项目clone到wsl系统的某文件夹内。
唤醒开发环境：在项目的根目录打开linux终端（WSL2），输入并运行：
code .
此时会唤起vscode并在wsl中打开该项目文件夹。
构建容器：在打开的vscode内安装 Dev Containers 插件
之后VSCode检测到容器配置文件，右下角会弹出的"在容器内重新打开"，点击即可进入容器。系统将依据Dockerfile执行环境下载和配置。 
等到终端可以输入后，说明环境已配置完成。
第二步：EasyEdit 框架源码回滚
实验使用的是旧版easyedit。
需要回滚至 EasyEdit 2023 年 10 月 1 日之前的稳定快照。
在容器的终端内执行：
Plain Text
# 完整克隆官方仓库到一个临时文件夹
git clone https://github.com/zjunlp/EasyEdit.git temp_repo
# 进入临时仓库
cd temp_repo
# 自动寻找 2023年10月1日之前的最后一次有效提交
SAFE_COMMIT=$(git rev-list -n 1 --before="2023-10-01" HEAD)
echo "系统锁定多模态污染前的安全 Commit ID 为: $SAFE_COMMIT"
# 将临时仓库的头指针强行回退到这一天
git checkout $SAFE_COMMIT
# 把框架源码和配置目录转移到主工作区
mv easyeditor ../
mv hparams ../
# 回到工作区根目录
cd ..
rm -rf temp_repo
第三步：下载模型 
运行download_models.py
即可下载TinyLlama-1.1B-Chat-v1.0到项目文件夹model内。
第四步：配置文件部署 (tinyllama.yaml)
在 hparams/ROME目录下新建 tinyllama.yaml，并配置以下参数以适配 TinyLlama 模型：
YAML
alg_name: "ROME"
model_name: "./models/TinyLlama-1.1B-Chat-v1.0"
stats_dir: "./data/stats/tinyllama"
device: 0
layers: [11]
fact_token: "subject_last"
v_num_grad_steps: 25
v_lr: 5e-1
v_loss_layer: 21 
v_weight_decay: 0.5
clamp_norm_factor: 4
kl_factor: 0.0625
mom2_adjustment: false
context_template_length_params: [[5, 10], [10, 10]]
rewrite_module_tmp: "model.layers.{}.mlp.down_proj"
layer_module_tmp: "model.layers.{}"
mlp_module_tmp: "model.layers.{}.mlp"
attn_module_tmp: "model.layers.{}.self_attn"
ln_f_module: "model.norm"
lm_head_module: "lm_head"
mom2_dataset: "wikipedia"
mom2_n_samples: 100000
mom2_dtype: "float32"
还要在hparams/MEMIT目录下新建tinyllama.yaml
YAML
alg_name: "MEMIT"
model_name: "./models/TinyLlama-1.1B-Chat-v1.0"
stats_dir: "./data/stats"
device: 0
layers: [4, 5, 6, 7, 8]
clamp_norm_factor: 0.75
layer_selection: "all"
fact_token: "subject_last"
v_num_grad_steps: 20
v_lr: 5e-1
v_loss_layer: 21
v_weight_decay: 0.5
kl_factor: 0.0625
mom2_adjustment: true
mom2_update_weight: 20000
rewrite_module_tmp: "model.layers.{}.mlp.down_proj"
layer_module_tmp: "model.layers.{}"
mlp_module_tmp: "model.layers.{}.mlp"
attn_module_tmp: "model.layers.{}.self_attn"
ln_f_module: "model.norm"
lm_head_module: "lm_head"
mom2_dataset: "wikitext"
mom2_n_samples: 100000
mom2_dtype: "float32"
batch_size: 16
max_length: 256
第五步：实验任务运行
确保 task 的源码与数据文件（.json）已置于项目根目录，现在就可以执行以下命令获得实验结果了：
Plain Text
# 运行任务 1：模型基准评测
python baseline.py
# 运行任务 2：单条事实编辑实践
python edit_rome.py
# 运行任务 3：批量知识编辑实践
python edit_memit.py
# 运行任务 4：综合评估 
python evaluate.py
其中，各task得到的结果都在输出终端中有中文提示，可以根据提示在项目文件夹内查看得到的json，图片，表格。

