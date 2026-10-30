// Keyframes (OpenRelight): the relit keyframes in one node.
// "Generate keyframes" (or ▶ on the node) runs the workflow up to here, no video. Each keyframe has a redraw button:
// it changes that keyframe's seed on Keyframe Seeds and runs up to here again, so only that one is redrawn.
// Run makes the video from the same keyframes: they are cached, not drawn again.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE = "OR_KeyframeReview";
const SEEDS = "OR_KeyframeSeeds";

const CSS = `
.orkf{font:12px system-ui,sans-serif;color:#ddd;display:flex;flex-direction:column;gap:6px;padding:4px}
.orkf .hint{color:#aaa;line-height:1.45}
.orkf .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:6px}
.orkf .card{background:#16181c;border:1px solid #333;border-radius:6px;overflow:hidden;display:flex;flex-direction:column}
.orkf .card img{width:100%;display:block;cursor:zoom-in;background:#000}
.orkf .card .bar{display:flex;align-items:center;justify-content:space-between;padding:4px 6px;gap:4px}
.orkf button{background:#2b2e35;color:#eee;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer}
.orkf button:hover{background:#3a3e48}.orkf button.main{background:#4a6bd8;border-color:#6d8cf0}
.orkf .empty{color:#777;padding:18px;text-align:center;border:1px dashed #444;border-radius:6px}
.orkf .big{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:10000;display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.orkf .big img{max-width:95vw;max-height:95vh}
`;

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
}

const viewURL = (c) => api.apiURL(`/view?filename=${encodeURIComponent(c.filename)}&type=${c.type || "temp"}` +
                                  `&subfolder=${encodeURIComponent(c.subfolder || "")}&t=${Date.now()}`);

// run the workflow only up to this node (the same as selecting it and pressing ▶). Queued by the node's id, not
// through the selection: "Queue Selected Output Nodes" says "nothing to queue" when the selection is not this node
// by the time it reads it (a click that lands on another node, Nodes 2.0 handling the click, 2026-10-30).
async function runUpTo(node) {
  app.canvas.deselectAll?.();
  app.canvas.selectNode(node);
  node._orkf?.start();
  const since = Date.now();
  const root = app.rootGraph ?? app.graph;
  if (node.graph && node.graph !== root)   // inside a subgraph: let the frontend work out its execution id
    await app.extensionManager.command.execute("Comfy.QueueSelectedOutputNodes");
  else
    await app.queuePrompt(0, 1, { queueNodeIds: [String(node.id), ...keyframeSavers(node)] });
  followRun(node, since);
}

// The image outputs (Save Image and the like) of this node's keyframes: fed by its outputs, or (older workflows) by
// the same images as its inputs. Run with it, they make the keyframes the job's result: without them the job's only
// images were the source clip's previews (Load Video, Trim Video), so the queue and the assets panel showed the
// original clip for a keyframes run (2026-10-30). Keyframes the plan does not make block their savers.
function keyframeSavers(node) {
  const g = node.graph || app.graph;
  const link = (id) => (id == null ? null : g.links?.[id] ?? g.links?.get?.(id));
  const from = new Set((node.inputs || []).filter((i) => i.type === "IMAGE").map((i) => link(i.link)).filter(Boolean)
    .map((l) => `${l.origin_id}:${l.origin_slot}`));
  const fed = (l) => l && (String(l.origin_id) === String(node.id) || from.has(`${l.origin_id}:${l.origin_slot}`));
  return (g._nodes || []).filter((n) => n !== node && n.mode === 0 && n.constructor?.nodeData?.output_node &&
    (n.inputs || []).some((i) => i.type === "IMAGE" && fed(link(i.link)))).map((n) => String(n.id));
}

// The run's keyframes read back from the history once the queue is empty. The "executed" message alone is not
// enough: after a page reload that reuses the client id, ComfyUI can drop this page's socket when the old socket's
// close arrives late, and the page gets no node results at all (01:52 run, 2026-10-30: keyframes made, panel empty).
async function followRun(node, since) {
  const id = String(node.id);
  const get = async (url) => (await api.fetchApi(url)).json();
  for (let i = 0; i < 1800; i++) {   // at most 1.5 hours
    await new Promise((r) => setTimeout(r, 3000));
    if (!node._orkf?.waiting) return;   // the "executed" message got here first
    let q;
    try { q = await get("/queue"); } catch (e) { continue; }
    if (q.queue_running?.length || q.queue_pending?.length) continue;
    let h = {};
    try { h = await get("/history?max_items=5"); } catch (e) { /* nothing to show */ }
    const run = Object.values(h).map((v) => ({ v, t: v.status?.messages?.[0]?.[1]?.timestamp || 0 }))
      .filter((r) => r.t >= since - 5000).sort((a, b) => b.t - a.t)[0]?.v;
    const text = run?.outputs?.[id]?.openrelight_keyframes?.[0];
    if (text) { node._orkf?.set(JSON.parse(text)); return; }
    const err = run?.status?.messages?.find((m) => m[0] === "execution_error")?.[1];
    node._orkf?.done(err ? `No keyframes were made: ${err.node_type}: ${String(err.exception_message || "").slice(0, 200)}`
      : run?.status?.messages?.some((m) => m[0] === "execution_interrupted") ? "Interrupted" : "");
    return;
  }
  node._orkf?.done("");
}

// the Keyframe Seeds node feeding keyframe i's sampler (there is one per workflow)
function seedsNode(node) {
  return (node.graph || app.graph)._nodes.find((n) => n.type === SEEDS);
}

class KeyframePanel {
  constructor(node, root) {
    this.node = node;
    this.root = root;
    this.cards = node.properties?.openrelight_keyframes || [];
    this.busy = new Set();
    if (!document.getElementById("orkf-css")) document.head.append(el("style", { id: "orkf-css" }, CSS));
    root.classList.add("orkf");
    this.render();
  }

  set(cards) {
    this.cards = cards;
    this.busy.clear();
    this.waiting = false;
    this.msg = "";
    this.node.properties = this.node.properties || {};
    this.node.properties.openrelight_keyframes = cards;
    this.render();
  }

  start() { this.waiting = true; this.msg = ""; this.render(); }

  done(msg) { this.busy.clear(); this.waiting = false; this.msg = msg; this.render(); }

  async redraw(i) {
    const seeds = seedsNode(this.node);
    const w = seeds?.widgets?.find((x) => x.name === `seed${i}`);
    if (!w) { alert("Keyframe Seeds node not found"); return; }
    w.value = (Number(w.value) + 1) % 2147483647;
    w.callback?.(w.value);
    this.busy.add(i);
    this.render();
    await runUpTo(this.node);
  }

  render() {
    const r = this.root;
    r.innerHTML = "";
    r.append(el("div", { class: "hint" },
      "1. Click Generate keyframes: this runs up to here and makes no video.",
      el("br"), "2. To try one again, click its Redraw button. Only that keyframe is redrawn.",
      el("br"), "3. When you are happy with them, click Run at the top to make the video from these keyframes."));
    r.append(el("div", {}, el("button", { class: "main", onclick: () => runUpTo(this.node), disabled: !!this.waiting },
      this.waiting ? "Generating… (shows here when done)" : "▶ Generate keyframes")));
    if (this.msg) r.append(el("div", { class: "hint", style: { color: "#ffb3a6" } }, this.msg));
    if (!this.cards.length) {
      r.append(el("div", { class: "empty" }, "No keyframes yet"));
      return;
    }
    const grid = el("div", { class: "grid" });
    for (const c of this.cards) {
      const img = el("img", { src: viewURL(c), title: "Click to enlarge" });
      img.onerror = () => { img.replaceWith(el("div", { class: "empty" }, "Preview cleared. Generate again to see it.")); };
      img.onclick = () => {
        const big = el("div", { class: "big", onclick: () => big.remove() }, el("img", { src: img.src }));
        r.append(big);
      };
      const label = `Keyframe ${c.index + 1}` + (c.frame != null ? ` · frame ${c.frame} (${c.sec}s)` : "");
      const busy = this.busy.has(c.index);
      grid.append(el("div", { class: "card" }, img, el("div", { class: "bar" },
        el("span", {}, label),
        el("button", { onclick: () => this.redraw(c.index), disabled: busy }, busy ? "Redrawing…" : "Redraw"))));
    }
    r.append(grid);
  }
}

api.addEventListener("executed", ({ detail }) => {
  const text = detail?.output?.openrelight_keyframes?.[0];
  if (!text || !app.graph) return;
  let cards;
  try { cards = JSON.parse(text); } catch (e) { return; }
  const id = String(detail.node).split(":").pop();
  const node = app.graph.getNodeById?.(id) ?? app.graph._nodes.find((n) => String(n.id) === id);
  node?._orkf?.set(cards);
});

app.registerExtension({
  name: "OpenRelight.Keyframes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      if (this.widgets?.some((w) => w.name === "keyframes")) return r;
      const root = document.createElement("div");
      root.style.width = "100%";
      const H = 420;
      const w = this.addDOMWidget("keyframes", "openrelightKeyframes", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => H, getMaxHeight: () => H, getHeight: () => H,
      });
      if (w) w.serialize = false;
      this.size = [Math.max(this.size?.[0] || 0, 620), this.size?.[1] || 0];
      this._orkf = new KeyframePanel(this, root);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this._orkf && this.properties?.openrelight_keyframes) this._orkf.set(this.properties.openrelight_keyframes);
      return r;
    };
  },
});
