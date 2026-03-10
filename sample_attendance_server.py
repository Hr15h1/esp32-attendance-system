"""
Campus Attendance Server
========================
• Trains a KNN classifier on face images stored in  dataset/<StudentName>/img1.jpg …
• Receives JPEG frames from ESP32-CAM via POST /recognize
• Marks attendance in a CSV + in-memory log
• Serves a live dashboard at GET /
• REST endpoints for the admin dashboard

Install:
    pip install flask flask-cors opencv-python face_recognition scikit-learn numpy pillow

Dataset layout:
    dataset/
        Alice_Johnson/
            1.jpg
            2.jpg
        Bob_Smith/
            1.jpg
"""

import os, io, csv, time, datetime, base64, pickle, logging
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

try:
    import face_recognition          # pip install face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False
    print("[WARN] face_recognition not installed — using mock recognizer")

from sklearn.neighbors import KNeighborsClassifier
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("dataset")
MODEL_PATH    = Path("knn_model.pkl")
ATTENDANCE_CSV = Path("attendance.csv")
CONFIDENCE_THRESHOLD = 0.55   # KNN distance threshold (lower = more strict)
COOLDOWN_SECONDS     = 60     # don't re-mark same person within 60 s
PORT = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── State ─────────────────────────────────────────────────────────────────────
knn_model    = None
label_names  = []
attendance_log = []        # list of dicts, in-memory
last_seen    = {}          # name → timestamp (cooldown)
latest_frame = None        # last JPEG bytes from ESP32

# ── Ensure CSV header ─────────────────────────────────────────────────────────
if not ATTENDANCE_CSV.exists():
    with open(ATTENDANCE_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time", "Confidence"])

# ─────────────────────────────────────────────────────────────────────────────
# KNN Training
# ─────────────────────────────────────────────────────────────────────────────
def train_knn():
    global knn_model, label_names

    if not DATASET_DIR.exists():
        log.warning("dataset/ folder not found — creating example structure")
        DATASET_DIR.mkdir()
        return False

    encodings, labels = [], []
    label_names = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])

    if not label_names:
        log.warning("No student folders in dataset/")
        return False

    log.info(f"Training KNN on {len(label_names)} students …")
    for label_idx, name in enumerate(label_names):
        folder = DATASET_DIR / name
        count  = 0
        for img_path in folder.glob("*.jpg"):
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                encodings.append(encs[0])
                labels.append(label_idx)
                count += 1
        for img_path in folder.glob("*.png"):
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                encodings.append(encs[0])
                labels.append(label_idx)
                count += 1
        log.info(f"  {name}: {count} face(s)")

    if not encodings:
        log.error("No face encodings extracted from dataset")
        return False

    n_neighbors = max(1, min(5, len(encodings) // len(label_names)))
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="ball_tree",
                                      metric="euclidean", weights="distance")
    knn_model.fit(encodings, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": knn_model, "labels": label_names}, f)
    log.info(f"✓ KNN trained ({len(encodings)} samples, k={n_neighbors}) → {MODEL_PATH}")
    return True


def load_or_train():
    global knn_model, label_names
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        knn_model   = data["model"]
        label_names = data["labels"]
        log.info(f"✓ Loaded KNN model ({len(label_names)} students)")
    elif USE_FACE_RECOGNITION:
        train_knn()


def recognize_face(frame_bytes: bytes):
    """Returns (name, confidence) or (None, 0)"""
    if not USE_FACE_RECOGNITION or knn_model is None:
        return "MockStudent", 0.99   # fallback for testing

    nparr = np.frombuffer(frame_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(img_rgb)
    if not face_locs:
        return None, 0

    enc = face_recognition.face_encodings(img_rgb, face_locs)[0]
    distances, indices = knn_model.kneighbors([enc], n_neighbors=1)
    dist = distances[0][0]
    idx  = indices[0][0]

    # Normalize distance to confidence (128-dim euclidean: ~0.4 same, ~0.8+ diff)
    confidence = max(0.0, 1.0 - dist / 0.8)

    if dist > CONFIDENCE_THRESHOLD * 0.8 + 0.3:   # loose gate
        return "Unknown", confidence

    name = label_names[idx]
    return name, round(confidence, 3)


def mark_attendance(name: str, confidence: float):
    now   = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    tstr  = now.strftime("%H:%M:%S")

    # Cooldown check
    key = f"{name}_{today}"
    if key in last_seen and (time.time() - last_seen[key]) < COOLDOWN_SECONDS:
        return False, "cooldown"

    last_seen[key] = time.time()

    record = {"name": name, "date": today, "time": tstr, "confidence": confidence}
    attendance_log.append(record)

    with open(ATTENDANCE_CSV, "a", newline="") as f:
        csv.writer(f).writerow([name, today, tstr, confidence])

    log.info(f"✅ Attendance marked: {name} @ {tstr} (conf={confidence})")
    return True, "marked"

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/recognize", methods=["POST"])
def recognize():
    """ESP32-CAM posts JPEG bytes here"""
    global latest_frame

    frame_bytes = None

    # Accept raw binary (Content-Type: image/jpeg)
    if request.content_type and "image" in request.content_type:
        frame_bytes = request.data
    # Accept JSON with base64 field
    elif request.is_json:
        data = request.get_json()
        frame_bytes = base64.b64decode(data.get("image", ""))
    # Accept multipart form
    elif "image" in request.files:
        frame_bytes = request.files["image"].read()
    else:
        return jsonify({"error": "No image data"}), 400

    latest_frame = frame_bytes   # store for live preview

    name, confidence = recognize_face(frame_bytes)

    if name is None:
        return jsonify({"status": "no_face", "name": None, "confidence": 0})

    if name == "Unknown":
        return jsonify({"status": "unknown", "name": "Unknown", "confidence": confidence})

    marked, reason = mark_attendance(name, confidence)
    return jsonify({
        "status":     "marked" if marked else reason,
        "name":       name,
        "confidence": confidence,
        "time":       datetime.datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/attendance", methods=["GET"])
def get_attendance():
    date_filter = request.args.get("date", datetime.date.today().isoformat())
    records = [r for r in attendance_log if r["date"] == date_filter]
    return jsonify({"date": date_filter, "count": len(records), "records": records})


@app.route("/attendance/all", methods=["GET"])
def get_all_attendance():
    return jsonify({"count": len(attendance_log), "records": attendance_log})


@app.route("/students", methods=["GET"])
def get_students():
    folders = []
    if DATASET_DIR.exists():
        for d in sorted(DATASET_DIR.iterdir()):
            if d.is_dir():
                imgs = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                folders.append({"name": d.name, "images": imgs})
    return jsonify({"students": folders, "model_ready": knn_model is not None})


@app.route("/train", methods=["POST"])
def retrain():
    if not USE_FACE_RECOGNITION:
        return jsonify({"status": "error", "message": "face_recognition not installed"})
    ok = train_knn()
    return jsonify({"status": "ok" if ok else "error",
                    "students": len(label_names)})


@app.route("/stream/latest", methods=["GET"])
def latest_image():
    """Return the most recent frame from ESP32 as JPEG"""
    if latest_frame is None:
        return "No frame yet", 204
    from flask import Response
    return Response(latest_frame, mimetype="image/jpeg")


@app.route("/upload_student", methods=["POST"])
def upload_student():
    """Upload a training image for a student"""
    name  = request.form.get("name", "").strip().replace(" ", "_")
    if not name or "image" not in request.files:
        return jsonify({"error": "name and image required"}), 400
    folder = DATASET_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("*.jpg")))
    path = folder / f"{existing+1}.jpg"
    request.files["image"].save(str(path))
    return jsonify({"status": "saved", "path": str(path)})


@app.route("/", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML (single-page, auto-refreshes)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Campus Attendance</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--accent:#00d4aa;--warn:#f59e0b;--red:#ef4444;--text:#e2e8f0;--muted:#64748b;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Space Mono',monospace;min-height:100vh;}
  header{padding:1.5rem 2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:1rem;}
  header h1{font-family:'Syne',sans-serif;font-weight:800;font-size:1.6rem;letter-spacing:-0.5px;}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
  .grid{display:grid;grid-template-columns:380px 1fr;gap:1.5rem;padding:1.5rem;height:calc(100vh - 70px);}
  .cam-panel{display:flex;flex-direction:column;gap:1rem;}
  #cam-feed{width:100%;aspect-ratio:4/3;border-radius:12px;border:1px solid var(--border);background:#000;object-fit:cover;}
  .result-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1.2rem;}
  .result-card .label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-bottom:.4rem;}
  .result-card .name{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:var(--accent);}
  .result-card .meta{font-size:.75rem;color:var(--muted);margin-top:.3rem;}
  .badge{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.7rem;font-weight:700;margin-top:.5rem;}
  .badge.marked{background:rgba(0,212,170,.15);color:var(--accent);}
  .badge.unknown{background:rgba(239,68,68,.15);color:var(--red);}
  .badge.cooldown{background:rgba(245,158,11,.15);color:var(--warn);}
  .right-panel{display:flex;flex-direction:column;gap:1rem;overflow:hidden;}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
  .stat-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:1rem;}
  .stat-card .val{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--accent);}
  .stat-card .lbl{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:.2rem;}
  .log-panel{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;}
  .log-header{padding:.9rem 1.2rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
  .log-header span{font-size:.65rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);}
  #log-body{flex:1;overflow-y:auto;padding:.5rem;}
  .log-row{display:grid;grid-template-columns:1fr 80px 80px 90px;gap:.5rem;padding:.6rem 1rem;border-radius:8px;font-size:.78rem;align-items:center;}
  .log-row:hover{background:var(--border);}
  .log-row .n{font-weight:700;color:var(--text);}
  .log-row .t{color:var(--muted);}
  .log-row .c{color:var(--accent);text-align:right;}
  .log-row .s{text-align:right;}
  #log-body::-webkit-scrollbar{width:4px;}
  #log-body::-webkit-scrollbar-track{background:transparent;}
  #log-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
  .no-data{color:var(--muted);font-size:.8rem;text-align:center;padding:2rem;}
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <h1>Campus Attendance System</h1>
  <span style="margin-left:auto;font-size:.7rem;color:var(--muted)" id="clock"></span>
</header>

<div class="grid">
  <div class="cam-panel">
    <img id="cam-feed" src="/stream/latest" alt="ESP32-CAM Feed" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22><rect fill=%22%23111827%22 width=%22400%22 height=%22300%22/><text x=%2250%%22 y=%2250%%22 font-family=%22monospace%22 font-size=%2214%22 fill=%22%2364748b%22 text-anchor=%22middle%22>Waiting for ESP32-CAM…</text></svg>'">
    <div class="result-card" id="result-card">
      <div class="label">Last Recognition</div>
      <div class="name" id="res-name">—</div>
      <div class="meta" id="res-meta">No scan yet</div>
      <span class="badge" id="res-badge"></span>
    </div>
  </div>

  <div class="right-panel">
    <div class="stats">
      <div class="stat-card"><div class="val" id="stat-today">0</div><div class="lbl">Present Today</div></div>
      <div class="stat-card"><div class="val" id="stat-total">0</div><div class="lbl">Total Records</div></div>
      <div class="stat-card"><div class="val" id="stat-students">0</div><div class="lbl">Registered Students</div></div>
    </div>

    <div class="log-panel">
      <div class="log-header">
        <span>Attendance Log — <span id="log-date"></span></span>
        <span id="last-update"></span>
      </div>
      <div id="log-body"><div class="no-data">No records yet today</div></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

function tick() {
  const now = new Date();
  $('clock').textContent = now.toLocaleTimeString();
  $('log-date').textContent = now.toLocaleDateString();
}
setInterval(tick, 1000); tick();

async function refreshFeed() {
  const img = $('cam-feed');
  img.src = '/stream/latest?' + Date.now();
}

async function refreshAttendance() {
  try {
    const today = new Date().toISOString().slice(0,10);
    const [att, all, stu] = await Promise.all([
      fetch('/attendance?date=' + today).then(r=>r.json()),
      fetch('/attendance/all').then(r=>r.json()),
      fetch('/students').then(r=>r.json()),
    ]);

    $('stat-today').textContent    = att.count;
    $('stat-total').textContent    = all.count;
    $('stat-students').textContent = stu.students.length;

    const body = $('log-body');
    if (att.records.length === 0) {
      body.innerHTML = '<div class="no-data">No records yet today</div>';
    } else {
      body.innerHTML = att.records.slice().reverse().map(r => `
        <div class="log-row">
          <span class="n">${r.name}</span>
          <span class="t">${r.time}</span>
          <span class="c">${(r.confidence*100).toFixed(1)}%</span>
          <span class="s"><span class="badge marked">✓ Present</span></span>
        </div>`).join('');
    }

    $('last-update').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

// Poll latest frame and result
async function pollLatest() {
  try {
    const r = await fetch('/attendance/all').then(r=>r.json());
    if (r.records.length > 0) {
      const last = r.records[r.records.length - 1];
      $('res-name').textContent = last.name;
      $('res-meta').textContent = last.date + ' at ' + last.time;
      const b = $('res-badge');
      b.textContent = '✓ Marked Present';
      b.className = 'badge marked';
    }
  } catch(e) {}
}

setInterval(refreshFeed, 2000);
setInterval(refreshAttendance, 3000);
setInterval(pollLatest, 3000);
refreshAttendance();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_or_train()
    log.info(f"\n🎓 Attendance Server → http://0.0.0.0:{PORT}")
    log.info("  POST /recognize       ← ESP32-CAM sends JPEG here")
    log.info("  GET  /attendance      ← today's attendance")
    log.info("  GET  /students        ← registered students")
    log.info("  POST /train           ← retrain KNN")
    log.info("  POST /upload_student  ← add training image\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
