# 波形叠影台 · 示波器分段、触发抖动与测量窗口谱系

面向信号完整性团队的复核工具：导入分段捕获、触发标记和通道标定，
对齐重复事件并审查测量分布。来源与解释分开保存，所有裁定可撤销、
可回溯、可导出重导。

## 安装与启动

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5361
```

打开 http://127.0.0.1:5361 ：

1. **载入仓库示例** —— 导入 `casepkg/wave-overlay-demo` 基线
   （每段原采样、时基、竖直增益、探头衰减、触发时标、deskew、测量区间版本化）。
2. **推进剧本** —— 依次经过：基线核对 → 参数修订（峰阈值 0.8→0.9）→
   区间版本修订（v1→v2）→ 局部源改动（ch2 deskew 350ps→300ps）→
   互相冲突的人工决定（后者遇版本冲突，落到 `br:review-b`，两者都保留）→
   第二分支 `br:edge-rules` 检验边界规则。
3. 在页面上完成一次分支裁定（接受/否定候选），可**撤销**；
   两个窗口同时保存时，后提交者看到版本冲突及双方差异；
   **导出档案 → 删库重导** 后身份、顺序、来源关系保持；
   事件历史里**回到此事件**可恢复当时的依赖版本与可见结论。

## 派生规则（机器可读判定事件）

- `nonuniform-no-index-sub`：时间轴不等间隔时不用索引相减，全部按真实时间坐标计算。
- `gap-no-interp`：缺样与饱和区间不进入插值与互相关。
- `multipeak-candidates`：对齐保留多峰互相关候选，无法唯一决定时保留全部合法候选。
- `trigger-at-boundary`：触发在边界的段被排除并记录原因。

参数/参考版本变化后按依赖关系计算失效范围（`derivations.stale`），
不全局悄悄替换旧结论；导入失败整体回滚，不留半份记录。

## 布局

- `app.py` — FastAPI 路由（页面 + JSON API）
- `waveforge/` — `units`（单位/坐标基准）、`ingest`（来源导入/诊断/回滚）、
  `derive`（对齐/抖动/测量窗）、`store`（事件日志/分支/失效范围）、
  `exporter`（导出/导入）、`scenario`（剧本回放）、`db`（schema + 迁移）
- `casepkg/wave-overlay-demo/` — 离线案例包（含反例档案 `counterexamples.json`）
- `tools/gen_casepkg.py` — 案例包生成器
- `tests/` — 单元、迁移、剧本/反例、导出往返、端到端流程
