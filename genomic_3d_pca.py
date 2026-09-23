"""
3D PCA visualization of patients in genomic space, colored by LUAD staging label.

Shows whether RNA-seq expression alone partially separates early-stage (I/II)
from late-stage (III/IV) LUAD patients — the biological motivation for including
genomic embeddings in the multimodal model.

Outputs:
  genomic_3d_pca.png       — static 3D scatter (for paper)
  genomic_3d_pca.html      — interactive 3D scatter (for STS presentation)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Load data ─────────────────────────────────────────────────────────────────
X    = np.load("combat_corrected_matrix.npy")
pids = np.load("combat_patient_ids.npy", allow_pickle=True).tolist()

# Load staging labels from the real patient CSV
df = pd.read_csv("real_patient_data.csv")
# One label per patient (first slice row is enough)
patient_labels = (
    df.drop_duplicates(subset="patient_id")
      .set_index("patient_id")["has_condition"]
)

labels = np.array([patient_labels.get(p, np.nan) for p in pids])
valid  = ~np.isnan(labels)
X_v    = X[valid]
labels_v = labels[valid].astype(int)
pids_v   = [p for p, ok in zip(pids, valid) if ok]

print(f"Patients with labels: {valid.sum()}  "
      f"(Stage III/IV: {labels_v.sum()}, Stage I/II: {(labels_v==0).sum()})")

# ── PCA ───────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_v)
pca = PCA(n_components=3, random_state=42)
coords = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100
print(f"PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  PC3={var[2]:.1f}%")

# ── Static matplotlib 3D scatter ──────────────────────────────────────────────
COLORS = {0: "#3498db", 1: "#e74c3c"}   # blue=early, red=late
STAGE  = {0: "Stage I/II (early)",  1: "Stage III/IV (late)"}
COHORT_MARKERS = {}   # filled in below

# Detect cohort from patient_id prefix for marker shape
def cohort(pid):
    if pid.startswith("TCGA"):   return "TCGA"
    if pid.startswith("C3"):     return "CPTAC"
    return "Stanford"

cohorts  = np.array([cohort(p) for p in pids_v])
markers  = {"TCGA": "o", "CPTAC": "^", "Stanford": "s"}

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

for lbl in [0, 1]:
    for coh, mk in markers.items():
        mask = (labels_v == lbl) & (cohorts == coh)
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=COLORS[lbl], marker=mk, s=40, alpha=0.72,
            edgecolors="white", linewidth=0.3,
            label=f"{STAGE[lbl]} · {coh}" if coh == "TCGA" else "_nolegend_"
        )

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[0], markersize=9,
           label="Stage I/II (early)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[1], markersize=9,
           label="Stage III/IV (late)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="grey", markersize=7, label="TCGA"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="grey", markersize=7, label="CPTAC"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="grey", markersize=7, label="Stanford"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.7)

ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=9, labelpad=6)
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=9, labelpad=6)
ax.set_zlabel(f"PC3 ({var[2]:.1f}%)", fontsize=9, labelpad=6)
ax.set_title(
    "Patients in RNA-seq PCA Space\n"
    "(ComBat-corrected, 18 514 genes → 3 PCs)",
    fontsize=11, fontweight="bold"
)
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig("genomic_3d_pca.png", dpi=150, bbox_inches="tight")
print("Saved → genomic_3d_pca.png")

# ── Interactive HTML (pure JS, no dependencies) ───────────────────────────────
def rgb_hex(c): return c  # already hex

pts_json = []
for i, (pid, lbl, coh) in enumerate(zip(pids_v, labels_v, cohorts)):
    pts_json.append({
        "x": round(float(coords[i,0]),4),
        "y": round(float(coords[i,1]),4),
        "z": round(float(coords[i,2]),4),
        "label": int(lbl),
        "cohort": coh,
        "pid": pid,
    })

import json
pts_str = json.dumps(pts_json)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LUAD Genomic PCA — 3D</title>
<style>
  body {{ margin:0; background:#0f1117; color:#eee; font-family:sans-serif; }}
  canvas {{ display:block; }}
  #info {{ position:absolute; top:10px; left:10px; font-size:13px; }}
  #tooltip {{ position:absolute; display:none; background:rgba(0,0,0,0.8);
              padding:6px 10px; border-radius:6px; font-size:12px; pointer-events:none; }}
  #legend {{ position:absolute; bottom:20px; left:20px; font-size:12px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">
  <b>LUAD Patients — RNA-seq PCA Space</b><br>
  <span style="color:#aaa">Drag to rotate · Scroll to zoom</span><br>
  PC1 {var[0]:.1f}%  PC2 {var[1]:.1f}%  PC3 {var[2]:.1f}%
</div>
<div id="tooltip"></div>
<div id="legend">
  <span class="dot" style="background:#3498db"></span>Stage I/II (early)&nbsp;&nbsp;
  <span class="dot" style="background:#e74c3c"></span>Stage III/IV (late)
</div>
<script>
const pts = {pts_str};
const canvas = document.getElementById('c');
const tooltip = document.getElementById('tooltip');
let W = window.innerWidth, H = window.innerHeight;
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext('2d');

// Camera state
let rotX = 0.3, rotY = 0.6, zoom = 180;
let dragging = false, lastX, lastY;

canvas.addEventListener('mousedown', e => {{ dragging=true; lastX=e.clientX; lastY=e.clientY; }});
window.addEventListener('mouseup', () => dragging=false);
window.addEventListener('mousemove', e => {{
  if(dragging) {{ rotY += (e.clientX-lastX)*0.01; rotX += (e.clientY-lastY)*0.01; lastX=e.clientX; lastY=e.clientY; draw(); }}
  else {{ checkHover(e.clientX, e.clientY); }}
}});
canvas.addEventListener('wheel', e => {{ zoom -= e.deltaY*0.3; zoom=Math.max(60,Math.min(500,zoom)); draw(); }});
window.addEventListener('resize', () => {{ W=window.innerWidth; H=window.innerHeight; canvas.width=W; canvas.height=H; draw(); }});

function project(x,y,z) {{
  const cosX=Math.cos(rotX), sinX=Math.sin(rotX);
  const cosY=Math.cos(rotY), sinY=Math.sin(rotY);
  let y2 = y*cosX - z*sinX, z2 = y*sinX + z*cosX;
  let x2 = x*cosY + z2*sinY; z2 = -x*sinY + z2*cosY;
  const fov=zoom, d=3.5; const s=fov/(d+z2*0.04);
  return {{ sx: W/2+x2*s, sy: H/2-y2*s, depth: z2 }};
}}

// normalise coords
const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y), zs=pts.map(p=>p.z);
const sc = 1/Math.max(...xs.map(Math.abs),...ys.map(Math.abs),...zs.map(Math.abs));

let projected = [];
function draw() {{
  ctx.fillStyle='#0f1117'; ctx.fillRect(0,0,W,H);
  projected = pts.map((p,i) => ({{...project(p.x*sc,p.y*sc,p.z*sc), i}}));
  projected.sort((a,b)=>a.depth-b.depth);
  for(const {{sx,sy,i}} of projected) {{
    const p=pts[i];
    const col = p.label===1 ? '#e74c3c' : '#3498db';
    ctx.beginPath(); ctx.arc(sx,sy,4,0,Math.PI*2);
    ctx.fillStyle=col+'cc'; ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.25)'; ctx.lineWidth=0.5; ctx.stroke();
  }}
}}

function checkHover(mx,my) {{
  let best=null, bd=16;
  for(const {{sx,sy,i}} of projected) {{
    const d=Math.hypot(sx-mx,sy-my);
    if(d<bd){{ bd=d; best=i; }}
  }}
  if(best!==null) {{
    const p=pts[best];
    const stg=p.label===1?'Stage III/IV':'Stage I/II';
    tooltip.style.display='block';
    tooltip.style.left=(mx+12)+'px'; tooltip.style.top=(my-10)+'px';
    tooltip.innerHTML=`<b>${{p.pid}}</b><br>${{stg}}<br>${{p.cohort}}`;
  }} else {{
    tooltip.style.display='none';
  }}
}}

draw();
</script>
</body>
</html>"""

with open("genomic_3d_pca.html", "w") as f:
    f.write(html)
print("Saved → genomic_3d_pca.html  (open in browser — drag to rotate, hover for patient info)")
