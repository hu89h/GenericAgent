# GADesktop 前端美化开发笔记本

**项目**：GADesktop-main-4
**仓库**：origin=dd3xp/GADesktop（上游），myfork=wjl2023/GADesktop（个人 fork）
**起始 commit**：4004590（与 origin/main 一致）
**主题**：前端美化（UI/UX/视觉优化）
**Bridge**：http://127.0.0.1:14169（PID 见 /tmp/bridge_main4.log）

---

## 0. 开发约束（每次开始前过一遍）

| # | 约束 | 说明 |
|---|------|------|
| 1 | 仅改前端 | 主战场：`frontends/desktop/static/` (app.js / index.html / styles.css / annot.* / vendor) |
| 2 | 零硬编码 | 颜色用 CSS 变量，文本走 i18n，禁止字面量 |
| 3 | 单文件优先 | 高内聚低耦合，能在一个文件解决就不跨文件 |
| 4 | 风格一致 | 与现有命名/缩进/注释风格保持一致 |
| 5 | 改前必读 | 用 file_read 看完目标段落再 patch |
| 6 | 小步 commit | 每个功能验证 OK 再 commit，标题用 `style(desktop): ...` 或 `feat(desktop): ...` |
| 7 | 大改起 subagent | 跨文件 / >100 行的改动起监察 |
| 8 | Cmd+Shift+R | 用户测试前提醒强制刷新清缓存 |

---

## 1. 项目入口速查

| 文件 | 作用 |
|------|------|
| `frontends/desktop_bridge.py` | aiohttp HTTP+WS 桥，端口 14168（默认），Web/桌面共用 |
| `frontends/desktop/static/index.html` | 主 HTML 框架 |
| `frontends/desktop/static/app.js` | 主前端逻辑（含 typewriter / 折叠 / scroll / 等） |
| `frontends/desktop/static/styles.css` | 主样式 |
| `frontends/desktop/static/annot.css/.js` | 注释相关 |
| `frontends/desktop/static/vendor/` | 第三方库 |

启动桥：
```bash
cd /Users/lwj/Documents/ga/GADesktop-main-4
BRIDGE_PORT=14169 nohup python3 frontends/desktop_bridge.py > /tmp/bridge_main4.log 2>&1 &
```

---

## 2. 美化方向（待用户填补）

> 本节随用户指示更新

- [ ] T1 — 待定
- [ ] T2 — 待定

---

## 3. 进度日志

| 日期 | 任务 | 文件 | 状态 | commit |
|------|------|------|------|--------|
| 2026-05-25 | 初始化笔记本，桥重启在 14169 | — | ✅ | — |
| 2026-05-25 | fix: summary 重复打印（最后一轮 head+body 双渲染） | app.js L826-838 | ✅ | pending |

---

## 4. 已知决策记录

（每次设计抉择记录于此，避免反复折腾）

- 主战场限定为 `frontends/desktop/static/`，不动 Python 后端
- CSS 变量优先于 inline style
- 文本走 i18n（看 index.html / app.js 中已有 i18n 机制再扩展）

---

## 5. PR 计划

- 分支名：`feat/frontend-beauty-v1`（待定）
- base：`origin/main` (4004590)
- 提交策略：每个美化任务一个 commit，最终汇总成一个 PR
- PR 描述模板见 `../memory/github_contribution_sop.md`

---

## 6. 反模式提醒（来自上次 PR#11 教训）

- ❌ 不读笔记直接动手 → 重复踩坑
- ❌ 一次改太多 → 改崩无法回退
- ❌ 硬编码颜色/文本 → 维护噩梦
- ❌ 跳过用户验证 → 问题累积
- ❌ 在崩溃代码上继续修 → 越修越烂，应 stash + checkout


---

## 7. Feature: 消息计时 (Message Timing)

**状态**：📋 计划中
**参考**：v1 (GenericAgent-main-6/frontends/genericagent-electron/renderer/app.js L292-415)
**目标**：在 assistant 消息上显示任务耗时 badge（实时跳动 → 结束后定格）

### 7.1 v1 核心逻辑摘要

| 函数 | 作用 |
|------|------|
| `formatDuration(ms)` | ms→"1ms"/"3.2s"/"2m 15s" |
| `formatTaskElapsed(ms)` | 带前缀格式化，用于 badge 文本 |
| `startTaskTimer(sess)` | runtime.taskStartedAt=now, setInterval(1s) 更新 live badge |
| `stopTaskTimer(sess)` | clearInterval, 计算 endedAt, badge 定格 |
| `ensureAssistantTaskElapsed(wrap, start, end)` | 在 .msg-wrap 上创建/更新 .task-elapsed badge |

CSS: `.task-elapsed { color: var(--accent); font-size: 12px; font-weight: 600; }`

### 7.2 v2 Hook 点分析

| 事件 | v2 位置 | 说明 |
|------|---------|------|
| 任务开始 | `setBusy(sess, true)` (L1451) | sendPrompt 内 |
| 任务结束 | `setBusy(sess, false)` (L1334/1343/1477) | poll 正常结束 / error |
| 流式 DOM | `r.draftEl` 创建 (L1094) | streaming 时的 assistant 消息 |
| 最终消息 | `appendMessage(sess, msg)` | poll 结束后正式渲染 |
| runtime 对象 | L896 | `{ polling, busy, lastId, seen, draftEl, draftText }` |

### 7.3 变更计划

**原则**：计时逻辑自包含为一个模块（~50行），通过 setBusy 的 2 个调用点接入，变化半径=2行。

| # | 变更 | 文件 | 位置 | 行数 |
|---|------|------|------|------|
| 1 | 添加 `formatTaskElapsed(ms)` 工具函数 | app.js | 工具函数区（formatDuration 不存在则新建） | ~15行 |
| 2 | runtime 增加 `taskStartedAt: null, taskTimerId: null` | app.js | L896 runtime 初始化 | +2字段 |
| 3 | 添加 `startTaskTimer(sess)` / `stopTaskTimer(sess)` | app.js | setBusy 上方 | ~25行 |
| 4 | setBusy 内调用：busy=true→start, busy=false→stop | app.js | L1144 setBusy 函数体 | +2行 |
| 5 | `ensureTaskElapsedBadge(wrap, startedAt, endedAt)` | app.js | 渲染辅助区 | ~15行 |
| 6 | draftEl 创建时挂 badge；appendMessage 对 role=assistant 挂 badge | app.js | L1094, L1043 | +2行 |
| 7 | `.task-elapsed` 样式 | styles.css | 消息样式区 | ~3行 |
| 8 | i18n key: `timing.elapsed` | app.js | i18n 对象 | +1行 |

**总改动**：~65 行新增，2 行修改。仅 app.js + styles.css。

### 7.4 关联影响

- `setBusy()` — 新增 2 行调用，不改原有逻辑
- `appendMessage()` — 在 assistant 分支末尾加 1 行 badge 挂载
- `draftEl` 创建处 — 加 1 行 badge 挂载
- 无 index.html 改动（badge 由 JS 动态创建）

### 7.5 风险

- 低：setInterval 泄漏 → stopTaskTimer 必须在所有 setBusy(false) 路径调用（已覆盖 3 处）
- 低：页面切换时 badge 更新 → 只更新当前 session 的 badge（用 data-session-id 过滤）

### 7.6 Bug修复：badge 闪烁 + 布局跳动

**现象**：badge 一跳一跳，只在变化时短暂显示，每次出现导致气泡上下跳动

**根因**：
1. `rewriteDraftBubble` 每 30ms 用 `innerHTML=` 重写 draftEl，badge 被销毁
2. timer 每 1000ms 才重建 badge → badge 只存活 30ms
3. CSS 无固定高度，badge 出现/消失改变容器高度

**修复**（3 处，仅 app.js + styles.css）：
1. `rewriteDraftBubble`：innerHTML 前保存 badge 文本，替换后恢复（L1133-1149）
2. `renderDraft`：创建 draftEl 时若正在计时立即挂载 badge（L1109）
3. CSS `.task-elapsed`：`display:block; min-height:18px; line-height:18px` 固定占位

**关联**：
- `rewriteDraftBubble` — 新增 badge 保存/恢复逻辑
- `renderDraft` — 新增 1 行 ensureTaskElapsedBadge 调用
- styles.css `.task-elapsed` — 改为 block + 固定行高

### 7.7 Code Review & PR

**PR #14**: https://github.com/dd3xp/GADesktop/pull/14
- 分支: `feat/message-timing-badge`
- 改动: app.js +93, styles.css +8
- Review 结论: 可合并，无阻塞问题
- 可优化点（后续迭代）：rewriteDraftBubble 可移动节点而非重建；多条 assistant 消息只有最后一条带 badge

---

## 8. 主题精简 + 气泡美化（进行中）

### TODO 清单

| # | 任务 | 目标 | 验证 | 状态 |
|---|------|------|------|------|
| 1 | 主题缩减为 2 个圆点 | 保留 swatch 圆点 UI，只留白色(浅色)和灰色(深色)两个 | 设置弹窗只显示 2 个圆点；点击切换 light/dark 正常；无控制台报错 | ✅ 已完成 |
| 2 | 气泡样式重构 | 待定 | 待定 | 待定 |
| 3 | 字体与行高优化 | 待定 | 待定 | 待定 |

---

### TODO 1: 主题缩减为 2 个圆点（白色=浅色，灰色=深色）

**改动计划**：
- **index.html**: `#theme-swatches` 从 8 个 swatch 缩减为 2 个
- **styles.css**: `--swatch-1`~`--swatch-8` 缩减为 2 个；删除多余 `html[data-theme]` 规则
- **app.js**: `applyTheme()` 简化为切换 light/dark appearance；swatch 点击逻辑适配


**实施完成** (2026-05-25)：
- index.html: 8 个 swatch 缩减为 2 个（data-theme="1" 白色, data-theme="2" 灰色）
- styles.css: 删除 swatch 3-8 颜色变量和规则；删除 theme="7"/"8" 残留规则
- app.js: applyTheme() 简化，swatch 与 appearance-seg 双向同步

**Bug 修复** (2026-05-25)：
- Bug1: 侧边栏选中项文字不可见 → 根因：`applyTheme` 中 `root.style.setProperty('--blue', swatch颜色)` 把全局强调色覆盖为白/灰 → 删除该行，--blue 保持 CSS 定义的 #1f6f53
- Bug2: swatch 高亮环永远在深色 → 同一根因，--blue 被设为白色导致 box-shadow 白底白环不可见 → 同上修复
- 额外：白色 swatch 的勾选标记改用深色描边（避免白底白勾不可见）

**验证结果**：
- --blue = #1f6f53（固定绿色强调色），不再被 swatch 颜色覆盖
- 侧边栏选中项文字 rgb(31,111,83) 在浅色/深色背景上均可见
- swatch 高亮环 rgb(31,111,83) 正确跟随选中状态切换

---

### 计时器刷新驻留修复 v2 (2026-05-25)

**问题：计时"已运行x"刷新后丢失**
- 根因：`r.taskStartedAt` 仅存于内存 runtime Map，刷新后 runtime 重建；pollSession 重新检测 running → setBusy → startTaskTimer 设新 Date.now()，真正开始时间丢失
- 修复方案（仅改 2 个函数，4 行新增）：
  1. `startTaskTimer`：尝试从 `sessionStorage.getItem('ga_task_started_' + sess.id)` 恢复时间戳，若无则用 Date.now()；设置后写入 sessionStorage
  2. `stopTaskTimer`：任务结束时 `sessionStorage.removeItem('ga_task_started_' + sess.id)`
- 为什么用 sessionStorage：与标签页生命周期绑定，关闭标签自动清除，不残留脏数据
- 为什么不需要 onBridgeReady 额外恢复：pollSession 检测到 running 后自动调 setBusy → startTaskTimer，此时 sessionStorage 中已有值会被读取恢复
- 改动文件：app.js（startTaskTimer L1198-1205 / stopTaskTimer L1219-1225）
