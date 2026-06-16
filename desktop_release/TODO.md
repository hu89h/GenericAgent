# GenericAgent Desktop 测试 TODO

## 目标

本轮测试只关注两个结果：

1. 当前系统能产出可用的环境配置脚本和桌面程序壳。
2. 在干净环境/干净仓库中，测试员可以用脚本完成环境配置，并成功运行这个程序壳。

不要求测试员填写复杂过程；只要证明结果可复现，并记录失败原因即可。

---

## 系统覆盖

重点测试 Windows 和 Linux；macOS 由项目负责人自测。

| 优先级 | 系统 | 架构/版本 | 测试员 | 状态 | 备注 |
|---|---|---|---|---|---|
| P0 | Windows 11 | x64 | 无 | 开发过程中已测试 | 当前最优先 Windows 环境 |
| P0 | Windows 10 | x64 | 杨航 | TODO | 常见 Windows 环境 |
| P0 | Ubuntu 24.04 LTS | x64 | 张景铭 | TODO | 当前常见 Linux LTS |
| P0 | Ubuntu 22.04 LTS | x64 | 张景铭 | TODO | 仍然常见的 Linux LTS |
| P1 | Debian 12 | x64 | 曹兮 | TODO | 可选，覆盖 Debian 系 |
| Owner | macOS | Apple Silicon 或 Intel | 杨航 | TODO |  |


---

## 每个系统需要完成的任务

### 任务 1：生成脚本和程序壳

测试员需要在自己的系统上产出：

- 对应系统的环境配置脚本；
- 对应系统的桌面程序壳（可执行程序）。

脚本实现可以直接参考 Windows 版本：`release/scripts/windows/install_windows.ps1`。核心做法是由脚本准备运行环境，并写入用户级桌面配置（如 `~/.ga_desktop_settings.json`），让程序壳能找到 `project_dir`、`python_path` 和 bridge 脚本。



### 任务 2：干净环境验证通过

从干净仓库或尽量干净的测试目录开始，验证：

- 环境配置脚本可以成功运行；
- 脚本能把运行桌面版需要的环境准备好；
- 从 `frontends/` 标准位置启动程序壳可以正常打开；
- 程序壳能连上本地 bridge/后端；

如果方便，建议用全新 clone 再走一遍；

### 任务 3：根据 CHECKLIST 测一下在不同平台下的界面使用兼容性 

---
