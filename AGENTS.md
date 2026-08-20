# AGENTS.md

## Language

- 默认使用中文回复，并称呼用户为 `nullptr`。
- English terms must include Chinese meaning on first use, for example GitHub（代码托管平台）, Git（版本控制系统）, branch（分支）, commit（提交）, and push（推送）。
- 不确定时必须明确说明不确定点，不要编造不存在的文件、接口、命令或结果。
- 生成的各种数据文件，png，csv，md等，尽量使用中文描述，如是英语专业名称，旁边需要附上中文解释。md文件也尽量用中文命名。
- 生成的md文件，log文件等命名前加上gpt-前缀，便于区分人工和AI生成的文件。
- md里的数学公式要渲染正确，不要使用\[
P_{\mathrm{cost}}=P_{\mathrm{el}}+P_{\mathrm{mech}}.
\],$r$这种无法渲染成功的格式。

## Environment

- 本机是 Ubuntu 22.04服务器，无GUI环境。gpu0和gpu1是A100，gpu2-5是V100。gpu5曾被标记为不可用；服务器于2026-08-17重启后，gpu5已通过CUDA矩阵运算、16环境50步Isaac Sim/PhysX检查及rough地形4096环境×2次更新容量冒烟，可用于后续任务，但启动前仍须与其他GPU一样检查占用和错误状态。我使用VSCode的SSH拓展远程连接。
- 除非我主动要求，否则不要用命令行帮我运行需要gpu的代码，所有需要gpu训练的代码运行必须由我本人在本机终端执行。只需要教我怎么使用命令行运行，我会在终端复制粘贴。现在已改用sh脚本运行。
- 对于不需要gpu训练的代码，可以在本机终端直接运行。例如查看log日志，ps等代码直接运行。
- 给出的命令行尽量不要用export，选定显卡应该在命令行中直接指定CUDA_VISIBLE_DEVICES，而不是在~/.bashrc或~/.zshrc中设置。
- 命令行有修改变动，应该跟我说明改动会影响什么，特别是会不会影响训练出来的模型权重。
- 超过 1000 次迭代的训练，必须在 tmux 或 GNU Screen
  等可断线恢复的终端会话中运行；相关 shell 脚本应拒绝在普通 SSH 前台终端直接启动。
- 正式评估不强制使用 tmux 或 GNU Screen，可以在普通终端直接运行。
- 无真实机器人等设备，目前只能做仿真。
- 对于长命令行使用单行命令行，不要使用换行符和反斜杠。

## GitHub Network

- 2026-08-20确认：服务器到GitHub的国际出口链路不稳定。直连HTTPS可以偶尔完成只读查询，但`git push`的数据上传请求会出现`HTTP 408 Request Timeout`；标准SSH 22端口不可用。这不是仓库权限错误，也不能据此判断GitHub全局故障。
- 当前Git配置通过`http://127.0.0.1:27897`访问GitHub。端口27897来自用户Windows电脑建立的SSH隧道；端口17897用于SSH连接本身。
- 任何`git fetch`、`git pull`、`git push`或GitHub API写操作前，先在服务器执行`ss -ltn | rg '127\.0\.0\.1:27897'`。如果没有监听，不要反复重试写操作，应请用户在Windows PowerShell运行`ssh -N -T xjy-clash-tunnel`并保持窗口运行；用户也提供了`ssh -N -T xjy-v2rayn-tunnel`作为备选隧道命令。
- 上述`ssh -N -T ...`命令必须在用户的Windows PowerShell执行，不是在Ubuntu服务器执行；相关SSH别名定义在用户本地配置中。
- 隧道可用时使用仓库的正常Git命令，让既有代理配置生效。不要在push时添加`-c http.https://github.com.proxy=`来绕过代理；服务器直连上传已被验证会超时。
- push输出若同时出现`HTTP 408`、`unexpected disconnect`和`Everything up-to-date`，不得把最后一行解释为成功。必须使用`git ls-remote --heads origin '<分支名>'`核对远端分支，并比较远端与本地完整commit哈希。
- 服务器当前没有可用于`git@github.com`的SSH公钥认证；不要擅自把remote改为SSH，也不要改写用户Git代理或凭据配置。
- 网络诊断不得输出访问令牌、Authorization头或`~/.config/gh/hosts.yml`内容。如确需curl/Git跟踪，必须启用脱敏并只报告HTTP状态、阶段和耗时。
