"""
Веб-интерфейс для генерации аккомпанемента к MIDI файлу.
Запуск: python app.py
Затем открой в браузере: http://localhost:5000
"""

import os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, request, jsonify, send_file, render_template_string
import torch
import numpy as np
import pretty_midi

from models.lstm_model import LSTMAccompanimentModel

FS              = 16
SEGMENT_STEPS   = 128
GUITAR_PROGRAMS = set(range(24, 32))
UPLOAD_DIR      = "outputs/uploads"
OUTPUT_DIR      = "outputs/midi"
DRUMS_MODEL     = "models/guitar_drums_final/model.pt"
PIANO_MODEL     = "models/guitar_piano_final/model.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

def load_model(path):
    model = LSTMAccompanimentModel(hidden_size=128, num_layers=1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model

drums_model = load_model(DRUMS_MODEL) if os.path.exists(DRUMS_MODEL) else None
piano_model = load_model(PIANO_MODEL) if os.path.exists(PIANO_MODEL) else None


def notes_to_roll(pm, T):
    roll = np.zeros((128, T), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if inst.program in GUITAR_PROGRAMS:
            for note in inst.notes:
                t0 = int(note.start * FS)
                t1 = max(t0 + 1, int(note.end * FS))
                if t0 < T:
                    roll[note.pitch, t0:min(t1, T)] = 1.0
    if roll.max() < 0.01:
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    t0 = int(note.start * FS)
                    t1 = max(t0 + 1, int(note.end * FS))
                    if t0 < T:
                        roll[note.pitch, t0:min(t1, T)] = 1.0
    return roll


def generate(input_roll, model):
    T = input_roll.shape[1]
    result = np.zeros((128, T), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    step = SEGMENT_STEPS // 2
    for start in range(0, T, step):
        seg = input_roll[:, start:start + SEGMENT_STEPS]
        if seg.shape[1] < SEGMENT_STEPS:
            seg = np.pad(seg, ((0, 0), (0, SEGMENT_STEPS - seg.shape[1])))
        x = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).squeeze().numpy()
        w = min(SEGMENT_STEPS, T - start)
        result[:, start:start + w] += pred[:, :w]
        counts[start:start + w] += 1
    return result / np.maximum(counts, 1)


def roll_to_instrument(roll, threshold, program, is_drum, name, max_simultaneous=None):
    """
    Конвертирует piano roll в MIDI инструмент.
    max_simultaneous — максимум нот одновременно (None = без ограничения).
    Для барабанов обычно 3-4, для пианино 2-3.
    """
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    dt = 1.0 / FS
    T = roll.shape[1]

    # Если задан лимит — в каждый момент оставляем только топ-N нот по вероятности
    if max_simultaneous is not None:
        filtered = np.zeros_like(roll)
        for t in range(T):
            col = roll[:, t]
            above = np.where(col > threshold)[0]
            if len(above) > max_simultaneous:
                # Оставляем только самые вероятные
                top = above[np.argsort(col[above])[-max_simultaneous:]]
                filtered[top, t] = col[top]
            else:
                filtered[above, t] = col[above]
        roll = filtered

    for pitch in range(128):
        active, start = False, 0.0
        for t in range(T):
            on = roll[pitch, t] > threshold
            if on and not active:
                start, active = t * dt, True
            elif not on and active:
                inst.notes.append(pretty_midi.Note(80, pitch, start, t * dt))
                active = False
        if active:
            inst.notes.append(pretty_midi.Note(80, pitch, start, T * dt))
    return inst


HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Генератор аккомпанемента</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f13; color: #e0e0e0; min-height: 100vh;
       display: flex; align-items: center; justify-content: center; padding: 20px; }
.card { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 16px;
        padding: 40px; width: 100%; max-width: 540px; }
h1 { font-size: 22px; font-weight: 600; color: #fff; margin-bottom: 6px; }
.subtitle { font-size: 13px; color: #555; margin-bottom: 28px; }

.drop-zone { border: 2px dashed #2a2a3a; border-radius: 12px; padding: 36px 20px;
             text-align: center; cursor: pointer; transition: all .2s; background: #13131c; }
.drop-zone:hover, .drop-zone.drag { border-color: #6c63ff; background: #16162a; }
.drop-zone .icon { font-size: 32px; margin-bottom: 10px; }
.drop-zone p { font-size: 13px; color: #666; }
.drop-zone strong { color: #999; }
.file-name { margin-top: 8px; font-size: 13px; color: #6c63ff; display: none; }

.section-title { font-size: 12px; color: #555; text-transform: uppercase;
                 letter-spacing: .08em; margin: 24px 0 12px; }

.option { display: flex; align-items: flex-start; gap: 12px; padding: 14px 16px;
          border: 1px solid #2a2a3a; border-radius: 10px; cursor: pointer;
          transition: all .2s; margin-bottom: 8px; }
.option:hover { border-color: #6c63ff; background: #16162a; }
.option.selected { border-color: #6c63ff; background: #16162a; }
.check { width: 18px; height: 18px; border: 2px solid #444; border-radius: 4px;
         display: flex; align-items: center; justify-content: center;
         flex-shrink: 0; margin-top: 1px; transition: all .2s; }
.option.selected .check { background: #6c63ff; border-color: #6c63ff; }
.check::after { content: '✓'; font-size: 11px; color: #fff; display: none; }
.option.selected .check::after { display: block; }
.opt-body { flex: 1; }
.opt-label { font-size: 14px; color: #ccc; margin-bottom: 2px; }
.opt-desc { font-size: 12px; color: #555; }

.controls { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
.ctrl-row { display: flex; align-items: center; gap: 10px;
            font-size: 12px; color: #666; }
.ctrl-row label { width: 100px; flex-shrink: 0; }
.slider { flex: 1; -webkit-appearance: none; height: 4px;
          background: #2a2a3a; border-radius: 2px; outline: none; }
.slider::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #6c63ff; cursor: pointer; }
.ctrl-val { font-size: 12px; color: #6c63ff; width: 32px; text-align: right; }
.ctrl-hint { display: flex; justify-content: space-between; font-size: 10px;
             color: #3a3a4a; margin-top: 2px; padding: 0 2px; }

.btn { width: 100%; padding: 14px; background: #6c63ff; color: #fff;
       border: none; border-radius: 10px; font-size: 15px; font-weight: 500;
       cursor: pointer; transition: all .2s; margin-top: 20px; }
.btn:hover:not(:disabled) { background: #5a52e0; }
.btn:disabled { background: #2a2a3a; color: #555; cursor: not-allowed; }

.status { margin-top: 16px; padding: 14px 16px; border-radius: 10px;
          font-size: 13px; display: none; }
.status.loading { background: #1a1a2e; border: 1px solid #2a2a4a; color: #888;
                  display: flex; align-items: center; gap: 8px; }
.status.success { background: #0d2018; border: 1px solid #1a4a30; color: #4caf82; display: block; }
.status.error   { background: #2a0d0d; border: 1px solid #4a1a1a; color: #cf6679; display: block; }

.download-btn { display: none; width: 100%; padding: 12px; background: #0d2018;
                color: #4caf82; border: 1px solid #1a4a30; border-radius: 10px;
                font-size: 14px; cursor: pointer; margin-top: 10px; text-align: center;
                text-decoration: none; }
.download-btn:hover { background: #102a1e; }

.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #333;
           border-top-color: #6c63ff; border-radius: 50%;
           animation: spin 0.8s linear infinite; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>🎵 Генератор аккомпанемента</h1>
  <p class="subtitle">Загрузи MIDI файл гитары и выбери что добавить</p>

  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="icon">🎸</div>
    <p><strong>Нажми или перетащи MIDI файл</strong></p>
    <p>.mid / .midi</p>
    <div class="file-name" id="fileName"></div>
  </div>
  <input type="file" id="fileInput" accept=".mid,.midi" style="display:none">

  <div class="section-title">Что генерировать</div>

  <div class="option" id="opt-drums" onclick="toggleOpt('drums')">
    <div class="check"></div>
    <div class="opt-body">
      <div class="opt-label">🥁 Барабаны</div>
      <div class="opt-desc">Ритм-паттерн на основе гитарной партии</div>
      <div class="controls" onclick="event.stopPropagation()">
        <div>
          <div class="ctrl-row">
            <label>Плотность нот</label>
            <input type="range" class="slider" id="drums-thr"
                   min="0.05" max="0.6" step="0.05" value="0.3"
                   oninput="upd('drums-thr','drums-thr-val')">
            <span class="ctrl-val" id="drums-thr-val">0.30</span>
          </div>
          <div class="ctrl-hint"><span>Редкие</span><span>Средние</span><span>Частые</span></div>
        </div>
        <div>
          <div class="ctrl-row">
            <label>Макс. нот сразу</label>
            <input type="range" class="slider" id="drums-max"
                   min="1" max="6" step="1" value="3"
                   oninput="upd('drums-max','drums-max-val')">
            <span class="ctrl-val" id="drums-max-val">3</span>
          </div>
          <div class="ctrl-hint"><span>1 (соло)</span><span>3 (норм)</span><span>6 (насыщ.)</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="option" id="opt-piano" onclick="toggleOpt('piano')">
    <div class="check"></div>
    <div class="opt-body">
      <div class="opt-label">🎹 Фортепиано</div>
      <div class="opt-desc">Гармонический аккомпанемент</div>
      <div class="controls" onclick="event.stopPropagation()">
        <div>
          <div class="ctrl-row">
            <label>Плотность нот</label>
            <input type="range" class="slider" id="piano-thr"
                   min="0.05" max="0.6" step="0.05" value="0.35"
                   oninput="upd('piano-thr','piano-thr-val')">
            <span class="ctrl-val" id="piano-thr-val">0.35</span>
          </div>
          <div class="ctrl-hint"><span>Редкие</span><span>Средние</span><span>Частые</span></div>
        </div>
        <div>
          <div class="ctrl-row">
            <label>Макс. нот сразу</label>
            <input type="range" class="slider" id="piano-max"
                   min="1" max="5" step="1" value="2"
                   oninput="upd('piano-max','piano-max-val')">
            <span class="ctrl-val" id="piano-max-val">2</span>
          </div>
          <div class="ctrl-hint"><span>1 (мелодия)</span><span>2-3 (аккорд)</span><span>5 (плотно)</span></div>
        </div>
      </div>
    </div>
  </div>

  <button class="btn" id="generateBtn" onclick="doGenerate()" disabled>Сгенерировать</button>
  <div class="status" id="status"></div>
  <a class="download-btn" id="downloadBtn">⬇ Скачать результат</a>
</div>

<script>
let selectedFile = null;
const selected = { drums: false, piano: false };

document.getElementById('dropZone').addEventListener('dragover', e => {
  e.preventDefault(); document.getElementById('dropZone').classList.add('drag'); });
document.getElementById('dropZone').addEventListener('dragleave', () =>
  document.getElementById('dropZone').classList.remove('drag'));
document.getElementById('dropZone').addEventListener('drop', e => {
  e.preventDefault(); document.getElementById('dropZone').classList.remove('drag');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]); });

function setFile(f) {
  selectedFile = f;
  const el = document.getElementById('fileName');
  el.textContent = '📄 ' + f.name; el.style.display = 'block';
  checkReady();
}
function toggleOpt(n) {
  selected[n] = !selected[n];
  document.getElementById('opt-' + n).classList.toggle('selected', selected[n]);
  checkReady();
}
function upd(sliderId, valId) {
  const v = document.getElementById(sliderId).value;
  const isInt = !v.includes('.');
  document.getElementById(valId).textContent = isInt ? v : parseFloat(v).toFixed(2);
}
function checkReady() {
  document.getElementById('generateBtn').disabled =
    !(selectedFile && (selected.drums || selected.piano));
}
async function doGenerate() {
  const btn = document.getElementById('generateBtn');
  const status = document.getElementById('status');
  const dlBtn = document.getElementById('downloadBtn');
  btn.disabled = true; dlBtn.style.display = 'none';
  status.className = 'status loading';
  status.innerHTML = '<span class="spinner"></span> Генерирую аккомпанемент...';
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('gen_drums', selected.drums ? '1' : '0');
  form.append('gen_piano', selected.piano ? '1' : '0');
  form.append('drums_threshold', document.getElementById('drums-thr').value);
  form.append('piano_threshold', document.getElementById('piano-thr').value);
  form.append('drums_max', document.getElementById('drums-max').value);
  form.append('piano_max', document.getElementById('piano-max').value);
  try {
    const res = await fetch('/generate', { method: 'POST', body: form });
    const data = await res.json();
    if (data.success) {
      status.className = 'status success';
      status.textContent = '✅ ' + data.message;
      dlBtn.href = '/download/' + data.filename;
      dlBtn.download = data.filename;
      dlBtn.style.display = 'block';
    } else {
      status.className = 'status error';
      status.textContent = '❌ ' + data.error;
    }
  } catch(e) {
    status.className = 'status error';
    status.textContent = '❌ Ошибка соединения с сервером';
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/generate', methods=['POST'])
def generate_route():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Файл не загружен"})
    f = request.files['file']
    if not f.filename.endswith(('.mid', '.midi')):
        return jsonify({"success": False, "error": "Нужен .mid или .midi файл"})

    gen_drums       = request.form.get('gen_drums', '0') == '1'
    gen_piano       = request.form.get('gen_piano', '0') == '1'
    drums_threshold = float(request.form.get('drums_threshold', 0.3))
    piano_threshold = float(request.form.get('piano_threshold', 0.35))
    drums_max       = int(request.form.get('drums_max', 3))
    piano_max       = int(request.form.get('piano_max', 2))

    if not gen_drums and not gen_piano:
        return jsonify({"success": False, "error": "Выбери хотя бы один инструмент"})

    uid      = str(uuid.uuid4())[:8]
    in_path  = os.path.join(UPLOAD_DIR, f"{uid}_input.mid")
    out_name = f"{uid}_result.mid"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    f.save(in_path)

    try:
        pm          = pretty_midi.PrettyMIDI(in_path)
        T           = int(pm.get_end_time() * FS) + 1
        guitar_roll = notes_to_roll(pm, T)
        pm_out      = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo())
        for inst in pm.instruments:
            pm_out.instruments.append(inst)

        info = []

        if gen_drums:
            if drums_model is None:
                return jsonify({"success": False, "error": "Модель барабанов не найдена"})
            roll = generate(guitar_roll, drums_model)
            inst = roll_to_instrument(roll, drums_threshold, 0, True,
                                      "Generated Drums", max_simultaneous=drums_max)
            pm_out.instruments.append(inst)
            info.append(f"барабаны: {len(inst.notes)} нот")

        if gen_piano:
            if piano_model is None:
                return jsonify({"success": False, "error": "Модель фортепиано не найдена"})
            roll = generate(guitar_roll, piano_model)
            inst = roll_to_instrument(roll, piano_threshold, 0, False,
                                      "Generated Piano", max_simultaneous=piano_max)
            pm_out.instruments.append(inst)
            info.append(f"пианино: {len(inst.notes)} нот")

        pm_out.write(out_path)
        return jsonify({"success": True, "filename": out_name, "message": ", ".join(info)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Файл не найден", 404


if __name__ == '__main__':
    print("Модель барабанов:", "✓" if drums_model else "✗ не найдена")
    print("Модель фортепиано:", "✓" if piano_model else "✗ не найдена")
    print("\nОткрой в браузере: http://localhost:5000\n")
    app.run(debug=False, port=5000)