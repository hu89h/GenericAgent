# Linux 桌面壳

真实二进制壳不直接提交到 Git 仓库。

测试时请从项目 Release/内部发布位置下载 Linux 版本壳，并放到本目录。

常见形式可能是：

```text
GenericAgent.AppImage
genericagent.deb
genericagent.rpm
```

具体文件名以后续发布说明为准。

## AppImage 构建参考

当前仓库尚未把 Linux AppImage 构建纳入正式 CI/workflow，也未将 `frontends/desktop/src-tauri/tauri.conf.json` 的 `bundle.targets` 改为固定包含 Linux target。

如需在 Linux 环境中临时构建 AppImage，可参考此前在 Ubuntu 24.04.4 LTS x64 VM 中初步跑通过的 Tauri CLI 命令：

```bash
cd frontends/desktop
npm install
npm run tauri build -- --bundles appimage
```

产物通常位于：

```text
frontends/desktop/src-tauri/target/release/bundle/appimage/*.AppImage
```

该流程目前仅作为后续正式 Linux 构建/CI 的参考，完整发布前仍需补充依赖安装、artifact 上传、校验和生成以及实际平台验证。
