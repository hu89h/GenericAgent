# Windows 桌面壳

真实二进制壳不直接提交到 Git 仓库。

测试时请从项目 Release/内部发布位置下载 Windows 版本壳，并放到本目录，例如：

```text
release/apps/windows/ga-desktop.exe
```

然后先运行：

```text
release/scripts/windows/install.bat
```

脚本完成后再启动桌面壳。