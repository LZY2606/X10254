"use strict";
const $ = (s) => document.querySelector(s);
let STATE = null, CASE = null;
let viewBranch = "main";
let viewingTip = null;        // historical event id (null = live tip)
let selectedCandidates = {};  // seg_id -> candidate_id (view choice, no write)

function toast(msg, ms = 5000) {
  const t = $("#toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(toast._h); toast._h = setTimeout(() => t.style.display = "none", ms);
}

async function api(path, opts = {}) {
  opts.headers = Object.assign({"Content-Type": "application/json"}, opts.headers || {});
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (r.status === 409) {
    const d = JSON.stringify(body.diff, null, 2);
    toast("409 版本冲突：\n" + d, 12000);
    throw new Error("conflict");
  }
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

async function loadState() {
  const params = new URLSearchParams({branch: viewBranch});
  STATE = await api("/api/state?" + params);
  CASE = await api("/api/case");
  viewingTip = null;
  render();
}

function tag(cls, text) {
  return `<span class="tag ${cls}" data-uid="tag:${text}">${text}</span>`;
}

function render() {
  if (!STATE) return;
  $("#branch-label").textContent = "branch " + STATE.branch;
  $("#tip-label").textContent = "tip " + (STATE.tip_event || "—").slice(0, 18);
  $("#window-now").innerHTML =
    `<span class="tag info" data-uid="window:version:${STATE.window.version}">` +
    `窗口 ${STATE.window.version}: ${STATE.window.start_repr} .. ${STATE.window.end_repr}</span>`;

  const ov = $("#img-overlay");
  ov.src = `/svg/overlay?branch=${encodeURIComponent(viewBranch)}&v=${STATE.tip_event||0}`;
  $("#img-jitter").src = `/svg/jitter?branch=${encodeURIComponent(viewBranch)}&v=${STATE.tip_event||0}`;

  renderBranches();
  renderSteps();
  renderSegments();
  renderTimeline();
}

function renderBranches() {
  const sel = $("#branch-select");
  sel.innerHTML = STATE.branches.map(b =>
    `<option value="${b.branch_id}" ${b.branch_id === viewBranch ? "selected" : ""}>` +
    `${b.label} (${b.branch_id.slice(0,14)})</option>`).join("");
}

function renderSteps() {
  const n = STATE.event_log.length;
  const steps = [
    {t: "导入并核对原始来源", d: "每段原采样、时基、竖直增益、探头衰减、触发时标、deskew、测量窗版本全部以原表示保存。",
     done: n >= 1},
    {t: "触发边界组合判定", d: "不等间隔时基 / 竖直饱和 / 多峰 / deskew / 缺样 / 触发在边界 均留下判定事件与受影响对象。",
     done: n >= 1},
    {t: "参数修订：测量窗 v1→v2", d: "依赖关系只令测量结论失效；历史事件仍可读回旧窗结果。",
     done: STATE.window.version === "v2"},
    {t: "只影响局部依赖的源数据改动", d: "补丁只令对应 segment/channel 派生失效，不全局替换。",
     done: STATE.event_log.some(e => e.type === "source-patched")},
    {t: "分支裁定（多峰候选）", d: "在 seg-04 接受/否定候选；旧分支与未采纳候选继续保留，可撤销。",
     done: !!STATE.rulings["seg-04-multipeak-alignment"]},
    {t: "制造并发冲突", d: "同基线两次保存窗口：后提交者看到 409 及双方差异。", done: false},
    {t: "最终复核路径", d: "叠影 → 抖动分布 → 触发候选 → 测量窗 → 被排除段原因；回看任一旧事件恢复当时结论。",
     done: true},
  ];
  $("#steps").innerHTML = steps.map((s, i) =>
    `<div class="step ${s.done ? "done" : ""}" data-uid="step:${i+1}">
       <b>${i+1}. ${s.t}</b><p>${s.d}</p>${s.done ? tag("ok","已留痕") : ""}
     </div>`).join("");
}

function sourceRows(seg, sm) {
  return ["CH1","CH2"].map(cid => {
    const q = sm[cid];
    return `<tr><td>${cid}</td>
      <td class="mono" data-uid="src:${seg.seg_id}:${cid}:timebase">${JSON.stringify(q.timebase_repr)}</td>
      <td class="mono" data-uid="src:${seg.seg_id}:${cid}:gain">${q.vertical_gain_repr}</td>
      <td class="mono" data-uid="src:${seg.seg_id}:${cid}:probe">${q.probe_attenuation_repr}</td>
      <td class="mono" data-uid="src:${seg.seg_id}:${cid}:deskew">${q.deskew_repr}</td>
      <td>${q.patched ? tag("warn","patched") : ""}</td></tr>`;
  }).join("");
}

function renderSegments() {
  $("#segments").innerHTML = STATE.segments.map(seg => {
    const j = seg.jitter || {};
    const cand = seg.candidates.map(c => {
      const sel = selectedCandidates[seg.seg_id] === c.candidate_id ||
                  c.adopted ? "sel" : "";
      return `<div class="cand ${sel}" data-uid="candidate:${c.candidate_id}">
        <label><input type="radio" name="pick:${seg.seg_id}"
          value="${c.candidate_id}" ${c.adopted ? "checked" : ""}
          onchange="pickCandidate('${seg.seg_id}','${c.candidate_id}')">
        <span class="mono">${c.candidate_id}</span></label>
        lag=${c.lag_ns.toFixed(3)} ns · score=${c.score.toFixed(3)} · rank ${c.rank}
        ${c.adopted ? tag("ok","已采纳") : ""}
        <span style="float:right">
          <button onclick="rule('${seg.seg_id}','${c.candidate_id}','accept')">接受</button>
          <button class="danger" onclick="rule('${seg.seg_id}','${c.candidate_id}','reject')">否定</button>
        </span></div>`;
    }).join("") || `<span class="muted">无合法候选（支持区触及缺样/饱和）</span>`;

    const measRows = Object.entries(seg.measurements).map(([cid, m]) =>
      `<tr data-uid="meas:${seg.seg_id}:${cid}:${m.window.version}">
         <td>${cid}</td><td>${m.window.version}</td>
         <td>${m.included ? tag("ok","included") : tag("excl", m.reason_code)}</td>
         <td>${m.vpp == null ? "—" : m.vpp.toFixed(4) + " V"}</td>
         <td>${(m.valid_fraction ?? 0).toFixed(2)}</td></tr>`).join("");

    const diags = seg.diagnostics.map(d => {
      const cls = d.kind === "decision" ? "info" :
                  d.code.includes("bad") || d.code.includes("saturation") ||
                  d.code.includes("gap") || d.code.includes("boundary") ? "warn" : "info";
      return tag(cls, d.code + (d.rule ? "@" + d.rule : ""));
    }).join(" ");

    return `<div class="panel" style="margin-bottom:10px"
             data-uid="segment:${seg.seg_id}">
      <b data-uid="segment:${seg.seg_id}:title">${seg.order}. ${seg.label}</b>
      <div class="mono" data-uid="segment:${seg.seg_id}:id">${seg.seg_id}</div>
      <div>${seg.included ? tag("ok","纳入测量/抖动") :
              tag("excl","排除: " + (seg.reason_code || ""))}</div>
      <img src="/svg/segment/${seg.seg_id}?branch=${encodeURIComponent(viewBranch)}&v=${STATE.tip_event||0}"
           data-uid="seg-svg:${seg.seg_id}"/>
      <details><summary>原始来源表示（导入保留，未规范化）</summary>
        <table><tr><th>ch</th><th>时基</th><th>竖直增益</th><th>探头衰减</th><th>deskew</th><th></th></tr>
        ${sourceRows(seg, seg.source_materials)}</table>
        <div class="mono" data-uid="src:${seg.seg_id}:trigger">
          trigger=${JSON.stringify(seg.trigger)}</div>
      </details>
      <p><b>触发候选</b>（多峰全部保留）</p>${cand}
      <p><b>测量窗</b></p>
      <table><tr><th>ch</th><th>窗</th><th>状态</th><th>Vpp</th><th>有效比</th></tr>${measRows}</table>
      <p><b>触发抖动</b>
        <span data-uid="jitter:${seg.seg_id}">${
          j.jitter_ns == null ? tag("excl", "排除: " + (j.exclusion_code||"—"))
          : (j.jitter_ns.toFixed(3) + " ns")}</span></p>
      <p><b>判定 / 决定事件</b><br>${diags || tag("ok","none")}</p>
    </div>`;
  }).join("");
}

function renderTimeline() {
  $("#timeline").innerHTML = STATE.event_log.map((e, i) =>
    `<div class="step" data-uid="event:${e.event_id}">
      <b>#${i+1} ${e.type}</b>
      <div class="mono">${e.event_id}</div>
      <button class="ghost" onclick="replayAt('${e.event_id}')">回看此时结论</button>
      ${i > 0 ? `<button class="ghost" onclick="undo('${e.event_id}')">撤销(开新分支)</button>` : ""}
     </div>`).join("");
}

window.pickCandidate = function(seg, cid) {
  selectedCandidates[seg] = cid;  // 只切换查看，不改原数据
  renderSegments();
};

async function mutate(fn) {
  try { await fn(); await loadState(); }
  catch (e) { if (e.message !== "conflict") toast("错误: " + e.message); }
}

window.rule = function(seg, cid, decision) {
  mutate(() => api("/api/rule", {method:"POST", body: JSON.stringify({
    segment: seg, candidate_id: cid, decision,
    branch: viewBranch, expected_last_event: STATE.tip_event,
    note: `reviewer ${decision}`})}));
};

window.undo = function(eventId) {
  mutate(async () => {
    const r = await api("/api/undo", {method:"POST", body: JSON.stringify({
      event_id: eventId, branch: viewBranch, label: "undo-" + Date.now()})});
    viewBranch = r.branch_id;
    toast("已开新分支恢复裁定前状态：" + r.branch_id.slice(0,14));
  });
};

window.replayAt = async function(eventId) {
  // fork a read-only-looking historical branch at that exact event
  const r = await api("/api/branch", {method:"POST", body: JSON.stringify({
    base_branch: viewBranch, label: "history@" + eventId.slice(0,8),
    at_event: eventId})});
  viewBranch = r.branch_id;
  await loadState();
  toast("已回到旧事件时点：派生结论使用当时的依赖版本（不是最新覆盖）");
};

$("#branch-select").addEventListener("change", async (e) => {
  viewBranch = e.target.value; await loadState();
});
$("#btn-refresh").onclick = () => { viewBranch = "main"; loadState(); };
$("#btn-load").onclick = async () => {
  try { await api("/api/load-bundled", {method:"POST"});
        toast("示例已导入"); await loadState(); }
  catch(e){ toast(e.message); }
};
$("#btn-win-v1").onclick = () => mutate(() => api("/api/window", {
  method:"POST", body: JSON.stringify({to_version:"v1", branch: viewBranch,
  expected_last_event: STATE.tip_event})}));
$("#btn-win-v2").onclick = () => mutate(() => api("/api/window", {
  method:"POST", body: JSON.stringify({to_version:"v2", branch: viewBranch,
  expected_last_event: STATE.tip_event})}));
$("#btn-patch").onclick = () => mutate(async () => {
  await api("/api/source-patch", {method:"POST", body: JSON.stringify({
    segment:"seg-08-window-revision", channel:"CH1",
    field:"reviewer_note", value: "local-only annotation " + Date.now(),
    branch: viewBranch, expected_last_event: STATE.tip_event})});
  toast("源数据补丁仅使 seg-08/CH1 派生失效（见事件 invalidation）");
});

$("#btn-conflict-a").onclick = async () => {
  window.__baseA = STATE.tip_event;
  try {
    await api("/api/window", {method:"POST", body: JSON.stringify({
      to_version:"v1", branch: viewBranch, expected_last_event: STATE.tip_event})});
    toast("审阅者A：v1 已保存"); await loadState();
  } catch(e){}
};
$("#btn-conflict-b").onclick = async () => {
  const base = window.__baseA;
  if (!base) { toast("请先点 审阅者A 保存 v1（记录共同基线）"); return; }
  try {
    await api("/api/window", {method:"POST", body: JSON.stringify({
      to_version:"v2", branch: viewBranch, expected_last_event: base})});
    toast("审阅者B：意外成功（不应发生）"); await loadState();
  } catch(e){}
};

$("#btn-export").onclick = async () => {
  const b = await api("/api/export");
  const blob = new Blob([JSON.stringify(b, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "waveform-lineage-export.json";
  a.click();
};
$("#btn-roundtrip").onclick = async () => {
  if (!confirm("删除本地 SQLite，再用导出档案导入；身份/顺序/来源关系应保持。")) return;
  const bundle = await api("/api/export");
  await api("/api/reset", {method:"POST"});
  const r = await api("/api/import", {method:"POST", body: JSON.stringify(bundle)});
  toast("再导入完成: " + JSON.stringify(r.imported)); await loadState();
};

loadState().catch(() => toast("尚未载入案例：点右上「载入仓库示例」"));
