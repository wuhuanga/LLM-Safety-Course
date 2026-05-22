​
EasyEdit实验复现方法​
第一步：启动容器与环境配置​
依赖 Docker 环境。需要宿主机已安装 Docker Desktop，并配置好 WSL2-ubuntu，开发工具是VScode。​
项目准备：将项目clone到wsl系统的某文件夹内。​
整个项目的目录如图：​
​
​
唤醒开发环境：在项目的根目录打开linux终端（WSL2），输入并运行：​
code .​
此时会唤起vscode并在wsl中打开该项目文件夹。​
构建容器：在打开的vscode内安装 Dev Containers 插件​
​
​
之后VSCode检测到容器配置文件，右下角会弹出的"在容器内重新打开"，点击即可进入容器。系统将依据Dockerfile执行环境下载和配置。 ​
等到终端可以输入后，说明环境已配置完成。​
第二步：EasyEdit 框架源码回滚​
实验使用的是旧版easyedit。​
需要回滚至 EasyEdit 2023 年 10 月 1 日之前的稳定快照。​
在容器的终端内执行：​
​
代码块​
# 完整克隆官方仓库到一个临时文件夹​
git clone https://github.com/zjunlp/EasyEdit.git temp_repo​
# 进入临时仓库​
cd temp_repo​
​

上传日志
联系客服
功能更新
帮助中心
效率指南
 
