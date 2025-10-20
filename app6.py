import torch
torch.set_num_threads(1)

from fastapi import FastAPI, Form, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import secrets
import html
from io import BytesIO
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import OpenAI
import docx
import re
import heapq

# ---------- Optional PDF/OCR libs (best-effort) ----------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:
        pdfminer_extract_text = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# =============================================================
# Configuration
# =============================================================
USERNAME = os.getenv("APP_USERNAME", "JayminShah")
PASSWORD = os.getenv("APP_PASSWORD", "Password1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "Jaymin123321/Rem-Classifier")

TOP_K = int(os.getenv("TOP_K", "20"))
LOW_MARGIN = float(os.getenv("LOW_MARGIN", "0.1"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))
CHUNK_CAP = int(os.getenv("CHUNK_CAP", "300"))
REM_KEYS = ("remuneration","compensation","pay","bonus","ltip","salary","incentive","director","executive")

# Flip labels switch
FLIP_LABELS = os.getenv("FLIP_LABELS", "1").strip() not in {"0", "false", "False", "no", "No"}

AGAINST_LABEL = 0
FOR_LABEL = 1

# =============================================================
# Models
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()
NUM_LABELS = classifier_model.config.num_labels

LABEL_MAP = {}
try:
    if getattr(classifier_model.config, "id2label", None):
        LABEL_MAP = {int(k): str(v).upper() for k, v in classifier_model.config.id2label.items()}
except Exception:
    LABEL_MAP = {}

FOR_INDEX, AGAINST_INDEX = 1, 0
if LABEL_MAP:
    for idx, label in LABEL_MAP.items():
        if "FOR" in label:
            FOR_INDEX = idx
        if "AGAINST" in label:
            AGAINST_INDEX = idx

print("Classifier num_labels:", NUM_LABELS)
print("Label map:", LABEL_MAP or "(none)")
print("Using FOR_INDEX:", FOR_INDEX, "AGAINST_INDEX:", AGAINST_INDEX)
print("FLIP_LABELS:", FLIP_LABELS)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
print("OpenAI client ready:", bool(client))

# =============================================================
# Data
# =============================================================
DF_PATH = os.getenv("POLICY_CSV", "investor_rem_policies.csv")
df = pd.read_csv(DF_PATH)
investor_policies = dict(zip(df["Investor"], df["RemunerationPolicy"]))

# ---------- CSV mapping + name matching helpers ----------
CSV_MAP = {
    "autotrader": os.getenv("AUTOTRADER_CSV", "autotrader_against_votes.csv"),
    "unilever": os.getenv("UNILEVER_CSV", "unilever_against_votes.csv"),
    "sainsbury": os.getenv("SAINSBURY_CSV", "sainsbury_against_votes.csv"),
    "leg": os.getenv("LEG_CSV", "leg_against_votes.csv"),
}

def _tokenize_name(s: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(s).lower()) if t]

def _prefix_key_from_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return " ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]

INVESTOR_PREFIX_INDEX: dict[str, set[str]] = {}
for inv_name in investor_policies.keys():
    toks = _tokenize_name(inv_name)
    keys = set()
    if toks:
        keys.add(toks[0])
        keys.add(_prefix_key_from_tokens(toks))
    for k in keys:
        if k:
            INVESTOR_PREFIX_INDEX.setdefault(k, set()).add(inv_name)

def _pick_manager_col(df_csv: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in df_csv.columns}
    candidates = [
        "vote manager", "manager", "votemanager",
        "investor", "investor name", "account", "organisation", "organization",
        "firm", "holder", "fund", "fund name"
    ]
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in df_csv.columns:
        if df_csv[c].dtype == object:
            return c
    return None

def _filter_against_rows(df_csv: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df_csv.columns}
    vote_candidates = ["vote", "decision", "voteresult", "vote result", "resolution vote", "voted"]
    for c in vote_candidates:
        if c in lower:
            col = lower[c]
            ser = df_csv[col].astype(str).str.lower()
            mask = ser.str.contains("against")
            if mask.any():
                return df_csv[mask]
            break
    return df_csv

def load_company_against_investors_from_csv(csv_path: str) -> set[str]:
    matched: set[str] = set()
    try:
        df_csv = pd.read_csv(csv_path)
    except Exception:
        return matched
    df_csv = _filter_against_rows(df_csv)
    manager_col = _pick_manager_col(df_csv)
    if not manager_col:
        return matched
    for raw_name in df_csv[manager_col].dropna().astype(str).tolist():
        toks = _tokenize_name(raw_name)
        key = _prefix_key_from_tokens(toks)
        tried = []
        if key:
            tried.append(key)
        if toks:
            tried.append(toks[0])
        for k in tried:
            invs = INVESTOR_PREFIX_INDEX.get(k)
            if invs:
                matched.update(invs)
                break
    return matched

# =============================================================
# FastAPI setup
# =============================================================
app = FastAPI()
sessions = {}
# Same-origin → CORS not needed; keep permissive but harmless
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def login_required(request: Request):
    token = request.cookies.get("session")
    if token and token in sessions:
        return sessions[token]
    return RedirectResponse(url="/login", status_code=302)

def escape_html(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")

# =============================================================
# File text extraction
# =============================================================
def extract_text_from_docx_bytes(data: bytes) -> str:
    document = docx.Document(BytesIO(data))
    paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in getattr(document, 'tables', []):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                paras.append("\t".join(cells))
    return "\n".join(paras)

def extract_text_from_pdf_bytes(data: bytes) -> str:
    if fitz is not None:
        try:
            text_parts = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            text = "\n".join(t for t in text_parts if t)
            if text and text.strip():
                return text
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            txt = pdfminer_extract_text(BytesIO(data))
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            out = []
            for page in reader.pages:
                out.append(page.extract_text() or "")
            txt = "\n".join(out)
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if pytesseract is not None and Image is not None and fitz is not None:
        try:
            out = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    out.append(pytesseract.image_to_string(img))
            return "\n".join(out)
        except Exception:
            pass
    raise RuntimeError("Unable to extract text from PDF. Install PyMuPDF or pdfminer.six for best results.")

# =============================================================
# Embeddings
# =============================================================
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return masked.sum(dim=1) / lengths

@torch.no_grad()
def get_embeddings(texts, batch_size: int = 64, max_length: int = 512):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = emb_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        outputs = emb_model(**enc)
        sent_emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
        all_vecs.append(sent_emb.cpu())
    return torch.cat(all_vecs, dim=0).numpy()

def get_embedding(text: str):
    return get_embeddings([text])[0]

# =============================================================
# Chunking
# =============================================================
def chunk_text(text: str, max_tokens: int = 512, stride: int = 256, min_tokens: int = 16):
    original_max = getattr(emb_tokenizer, "model_max_length", 512)
    try:
        emb_tokenizer.model_max_length = 10**9
        ids = emb_tokenizer.encode(text, add_special_tokens=False, truncation=False)
    finally:
        emb_tokenizer.model_max_length = original_max

    chunks = []
    for start in range(0, len(ids), stride):
        window = ids[start:start+max_tokens]
        if len(window) < min_tokens:
            continue
        chunk = emb_tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(chunk)
        if start + max_tokens >= len(ids):
            break
    return chunks

# =============================================================
# Classifier helpers
# =============================================================
@torch.no_grad()
def batch_predict_votes(policy: str, chunk_list: list[str], max_length: int = 512):
    # encode policy once
    p = cls_tokenizer(policy, truncation=True, max_length=max_length//2, add_special_tokens=False)
    p_ids = p["input_ids"]

    ids_batch, tti_batch, am_batch = [], [], []
    pad_id = cls_tokenizer.pad_token_id or 0

    for c in chunk_list:
        c_enc = cls_tokenizer(c, truncation=True, max_length=max_length//2, add_special_tokens=False)
        ids = cls_tokenizer.build_inputs_with_special_tokens(p_ids, c_enc["input_ids"])
        tti = cls_tokenizer.create_token_type_ids_from_sequences(p_ids, c_enc["input_ids"])
        if len(ids) > max_length:
            ids = ids[:max_length]
            tti = tti[:max_length]
        ids_batch.append(ids)
        tti_batch.append(tti)
        am_batch.append([1] * len(ids))

    L = max(len(x) for x in ids_batch)
    def pad(seq, val): return seq + [val]*(L - len(seq))

    input_ids = torch.tensor([pad(x, pad_id) for x in ids_batch], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([pad(x, 0) for x in tti_batch], dtype=torch.long, device=device)
    attention_mask = torch.tensor([pad(x, 0) for x in am_batch], dtype=torch.long, device=device)

    logits = classifier_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    ).logits  # [B, C]

    if logits.ndim == 1 or logits.size(-1) == 1:
        probs_against = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        pred_labels = (probs_against >= 0.5).astype(int)
        pred_labels = np.where(pred_labels == 1, AGAINST_LABEL, FOR_LABEL)
    else:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_against = probs[:, AGAINST_INDEX]
        probs_for = probs[:, FOR_INDEX]
        pred_labels = np.where(probs_against >= probs_for, AGAINST_LABEL, FOR_LABEL)

    if FLIP_LABELS:
        pred_labels = np.where(pred_labels == AGAINST_LABEL, FOR_LABEL, AGAINST_LABEL)
        probs_against = 1.0 - probs_against

    return pred_labels.tolist(), probs_against.tolist()

# Streaming top-K: no full N×D matrix in RAM
def topk_chunks_by_sim(chunks, policy_emb, k=TOP_K, batch_size=32):
    heap = []  # (sim, global_idx)
    idx_offset = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embs = get_embeddings(batch, batch_size=batch_size)  # [B, D]
        sims = batch_embs @ policy_emb  # [B]
        for j, s in enumerate(sims):
            gidx = idx_offset + j
            s = float(s)
            if len(heap) < k:
                heapq.heappush(heap, (s, gidx))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, gidx))
        idx_offset += len(batch)
    heap.sort(reverse=True)
    top_idx = [g for (s, g) in heap]
    top_sims = [s for (s, g) in heap]
    return [chunks[i] for i in top_idx], np.array(top_sims, dtype=np.float32)

# =============================================================
# Precompute investor embeddings (big speedup)
# =============================================================
print("Precomputing investor embeddings...")
INVESTOR_EMBS: dict[str, np.ndarray] = {}
with torch.no_grad():
    if investor_policies:
        names, texts = zip(*investor_policies.items())
        vecs = get_embeddings(list(texts), batch_size=32)
        for n, v in zip(names, vecs):
            INVESTOR_EMBS[n] = v
print(f"Cached {len(INVESTOR_EMBS)} investor embeddings.")

# =============================================================
# Routes
# =============================================================
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html><body>
    <h2>Login</h2>
    <form method='post' action='/login'>
        <label>Username: <input type='text' name='username'></label><br><br>
        <label>Password: <input type='password' name='password'></label><br><br>
        <input type='submit' value='Login'>
    </form>
    </body></html>
    """

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, USERNAME) and secrets.compare_digest(password, PASSWORD):
        token = secrets.token_urlsafe(16)
        sessions[token] = username
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            secure=True,   # HTTPS on Fly
            samesite="lax" # same-origin; use "none" if different domains
        )
        return response
    return HTMLResponse("<h3>Invalid credentials. <a href='/login'>Try again</a></h3>", status_code=401)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = login_required(request)
    if isinstance(user, Response):
        return user

    options_html = "".join(
        f"<option value=\"{html.escape(inv)}\">{html.escape(inv)}</option>"
        for inv in investor_policies.keys()
    )

    return """
    <html><body>
    <h2>Investor Vote Explanation Tool</h2>
    <form id='analyzeForm' action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file' accept='.docx,.pdf'><br><br>
        <label>Select Investor:</label><br>
        <select name='policy'>
            <option value='all'>All</option>
            %s
        </select><br><br>
        <input type='submit' value='Analyze'>
    </form>

    <div id='loader' style='display:none;'>⏳ Analyzing...</div>
    <div id='results'></div>

    <br>
    <button type="button" onclick="exportCSV()">📄 Export CSV</button>

    <script>
      function toCSVCell(v) {
        if (v == null) return '';
        const s = String(v).replaceAll('"', '""');
        return '"' + s + '"';
      }

      function exportCSV() {
        try {
          const blocks = document.querySelectorAll('.result-block');
          if (!blocks || blocks.length === 0) {
            alert('No results to export yet. Run an analysis first.');
            return;
          }
          const rows = [['Investor','Verdict']];
          blocks.forEach(block => {
            const investor = block.getAttribute('data-investor') || '';
            const verdict  = block.getAttribute('data-verdict')  || '';
            rows.push([investor, verdict]);
          });
          const csv = '\\ufeff' + rows.map(r => r.map(toCSVCell).join(',')).join('\\n');
          const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
          const url  = URL.createObjectURL(blob);
          const a    = document.createElement('a');
          a.href = url;
          a.download = 'analysis_results.csv';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(() => URL.revokeObjectURL(url), 500);
        } catch (e) {
          console.error('Export failed:', e);
          alert('Export failed. Check the console for details.');
        }
      }
    </script>

    <script>
    document.getElementById('analyzeForm').onsubmit = async function(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        document.getElementById('loader').style.display = 'block';
        document.getElementById('results').innerHTML = '';
        const response = await fetch('/upload', { method: 'POST', body: formData, credentials: 'include' });
        if (!response.body) {
          const txt = await response.text();
          document.getElementById('results').innerHTML = '<pre>' + txt + '</pre>';
          document.getElementById('loader').style.display = 'none';
          return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            document.getElementById('results').innerHTML += chunk;
        }
        document.getElementById('loader').style.display = 'none';
    };
    </script>

    </body></html>
    """ % options_html


@app.post("/upload", response_class=StreamingResponse)
def upload_file(request: Request, file: UploadFile = File(...), policy: str = Form(...)):
    user = login_required(request)
    if isinstance(user, Response):
        return user

    contents = file.file.read()
    filename = (file.filename or "").lower()

    # infer company from filename
    base = os.path.splitext(os.path.basename(filename))[0]
    company_key = None
    if "autotrader" in base:
        company_key = "autotrader"
    elif "unilever" in base:
        company_key = "unilever"
    elif "leg" in base:
        company_key = "leg"
    elif "sainsbury" in base or "sainsbury's" in base or "j sainsbury" in base:
        company_key = "sainsbury"

    csv_force_reason_investors: set[str] = set()
    if company_key:
        csv_path = CSV_MAP.get(company_key)
        if csv_path and os.path.exists(csv_path):
            try:
                csv_force_reason_investors = load_company_against_investors_from_csv(csv_path)
                print(f"[CSV] Matched {len(csv_force_reason_investors)} investors from {csv_path}")
            except Exception as _e:
                print(f"[CSV] Failed to load {csv_path}: {_e}")
        else:
            print(f"[CSV] No CSV available or path missing for company '{company_key}'")

    def stream():
        # --- Extract text ---
        try:
            if filename.endswith(".docx"):
                full_text = extract_text_from_docx_bytes(contents)
            elif filename.endswith(".pdf"):
                full_text = extract_text_from_pdf_bytes(contents)
            else:
                yield f"<p>Unsupported file type: {html.escape(filename)}. Please upload .docx or .pdf.</p>"
                return
        except Exception as e:
            yield f"<p>Error extracting text: {escape_html(str(e))}</p>"
            return

        if not full_text.strip():
            yield "<p>No readable text found in document.</p>"
            return

        yield "<p>✅ Text extracted.</p>"

        # --- Chunking + prefilter/cap ---
        yield "<p>✂️ Chunking…</p>"
        chunks = chunk_text(full_text)
        if len(chunks) > CHUNK_CAP:
            pri = [c for c in chunks if any(k in c.lower() for k in REM_KEYS)]
            if len(pri) >= CHUNK_CAP:
                chunks = pri[:CHUNK_CAP]
            else:
                tail_needed = CHUNK_CAP - len(pri)
                # deterministic "rest": keep order, skip duplicates
                seen = set(map(id, pri))
                rest = []
                for c in chunks:
                    if id(c) in seen:
                        continue
                    rest.append(c)
                    if len(rest) >= tail_needed:
                        break
                chunks = pri + rest
        if not chunks:
            yield "<p>Document is too short to chunk.</p>"
            return
        yield f"<p>📦 Using {len(chunks)} chunks.</p>"

        # --- Selection + classification ---
        label = "all" if policy.lower() == "all" else policy
        yield f"<p>⚙️ Computing embeddings & classifying for {html.escape(label)}…</p>"

        def analyze_investor(name: str, investor_policy: str, force_reason=False):
            # cached embedding
            policy_emb = INVESTOR_EMBS.get(name) or get_embedding(investor_policy)

            # top-K without full matrix; smaller batch keeps server responsive
            top_chunks, top_sims = topk_chunks_by_sim(chunks, policy_emb, k=TOP_K, batch_size=32)

            # one batched classifier forward for the K pairs
            preds, probs_against = batch_predict_votes(investor_policy, top_chunks)
            scored = [(top_chunks[i], int(preds[i]), float(probs_against[i])) for i in range(len(top_chunks))]
            maj, conf, frac, mean_prob = weighted_decision(scored, top_sims)

            maj_display = AGAINST_LABEL if bool(force_reason) else maj
            verdict = "AGAINST" if maj_display == AGAINST_LABEL else "FOR"

            need_reason = (maj_display == AGAINST_LABEL)
            reason_html = ""
            if need_reason:
                if client is None or not OPENAI_API_KEY:
                    reason_text = "OpenAI key not set — set OPENAI_API_KEY to see reasons"
                else:
                    top_chunk_texts = [c for c, _, _ in scored]
                    gpt_text = get_gpt_reason(investor_policy, top_chunk_texts)
                    reason_text = gpt_text or "(No explanation returned)"
                reason_html = (
                    "<div style='background:#f7f7f7;padding:10px;border-left:4px solid #cc0000;"
                    "margin-top:6px;color:#333;'><b>Reason:</b><br>" + escape_html(reason_text) + "</div>"
                )

            yield (
                f"<div class='result-block' data-investor='{html.escape(name, quote=True)}' "
                f"data-verdict='{html.escape(verdict, quote=True)}'>"
                f"<h3>Investor: {html.escape(name)}</h3>"
                f"<h4>{'❌ AGAINST' if maj_display == AGAINST_LABEL else '✅ FOR'}</h4>"
                f"{reason_html}"
                f"<hr></div>"
            )

        # Iterate investors
        if policy.lower() == "all":
            inv_list = list(investor_policies.items())
            total = len(inv_list)
            yield f"<p>👥 {total} investors to process…</p>"
            for idx, (inv, pol) in enumerate(inv_list, 1):
                yield f"<p>➡️ {idx}/{total}: {escape_html(inv)}</p>"
                yield from analyze_investor(inv, pol, force_reason=(inv in csv_force_reason_investors))
        else:
            pol = investor_policies.get(policy)
            if not pol:
                yield f"<p>Unknown investor {html.escape(policy)}</p>"
                return
            yield f"<p>🔹 Running selection & classification for {html.escape(policy)}…</p>"
            yield from analyze_investor(policy, pol, force_reason=(policy in csv_force_reason_investors))

    return StreamingResponse(
        stream(),
        media_type="text/html",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
