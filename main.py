import os
import re
import io
import json
import threading
import traceback
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

# CRITICAL FOR WINDOWS + PADDLE 3.x: disable oneDNN / PIR C++ crash before importing paddle
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
except ImportError:
    Image = None

# ----------------- OCR engine selection -----------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").strip().lower()
if OCR_ENGINE not in ("paddle", "tesseract"):
    print(f"[OCR] Unknown OCR_ENGINE '{OCR_ENGINE}' - using 'paddle'")
    OCR_ENGINE = "paddle"

PADDLE_LANG = os.getenv("PADDLE_OCR_LANG", "en")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
PADDLE_DOC_ORIENTATION = os.getenv("PADDLE_DOC_ORIENTATION", "0") == "1"
# Keep MKLDNN False on Windows to avoid NotImplementedError in onednn_instruction.cc
PADDLE_ENABLE_MKLDNN = os.getenv("PADDLE_ENABLE_MKLDNN", "0") == "1"
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "6"))

OCR_AVAILABLE = False
OCR_INIT_ERROR = None

if OCR_ENGINE == "tesseract":
    try:
        import pytesseract
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        OCR_AVAILABLE = True
    except ImportError as e:
        OCR_INIT_ERROR = e
else:
    try:
        from paddleocr import PaddleOCR
        OCR_AVAILABLE = True
    except Exception as e:
        OCR_INIT_ERROR = e

print(f"[OCR] engine={OCR_ENGINE}  available={OCR_AVAILABLE}" + (f"  error={OCR_INIT_ERROR!r}" if OCR_INIT_ERROR else ""))

try:
    from PyPDF2 import PdfReader
    PDF_TEXT_AVAILABLE = True
except ImportError:
    PDF_TEXT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False

SECRET_KEY = "bharat-vault-secure-token-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
UPLOAD_DIR = "uploads"
DB_FILE = "terra_digitize.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

app = FastAPI(title="BharatVault API", version="8.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR_CACHE: Dict[str, Tuple[str, float]] = {}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Pending Verification',
        raw_text TEXT,
        owner_name TEXT,
        seller_name TEXT,
        buyer_name TEXT,
        share_hissa TEXT,
        survey_number TEXT,
        khasra_number TEXT,
        khata_number TEXT,
        plot_number TEXT,
        property_id TEXT,
        total_area REAL,
        sub_plot_areas TEXT,
        land_use TEXT,
        boundaries_desc TEXT,
        village TEXT,
        tehsil TEXT,
        district TEXT,
        registration_date TEXT,
        mutation_date TEXT,
        serial_number TEXT,
        consideration_amount REAL,
        confidence_scores TEXT,
        validation_flags TEXT,
        document_type TEXT,
        uploaded_by TEXT,
        ocr_quality REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

    def _safe_add_column(table, coldef):
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    _safe_add_column("users", "phone_number TEXT")
    _safe_add_column("records", "submission_source TEXT DEFAULT 'Officer'")
    _safe_add_column("records", "submitted_by_user_id INTEGER")
    _safe_add_column("records", "verification_deadline TEXT")
    _safe_add_column("records", "escalated INTEGER DEFAULT 0")
    _safe_add_column("records", "has_conflict INTEGER DEFAULT 0")
    _safe_add_column("records", "verified_by TEXT")
    _safe_add_column("records", "verified_at TEXT")
    _safe_add_column("records", "verification_checklist TEXT")
    _safe_add_column("records", "verification_notes TEXT")

    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
            ("admin", "System Administrator", hash_password("admin123"), "Admin")
        )
        conn.commit()
    conn.close()

init_db()

PUBLIC_VERIFICATION_SLA_DAYS = 10
PUBLIC_AWAITING_STATUS = "Awaiting Physical Verification"

def log_action(conn: sqlite3.Connection, username: str, role: str, action: str, details: str = ""):
    conn.execute(
        "INSERT INTO audit_logs (user_id, role, action, details) VALUES (?, ?, ?, ?)",
        (username, role, action, details)
    )
    conn.commit()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    conn: sqlite3.Connection = Depends(get_db)
):
    actual_token = token or token_query
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session invalid or expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not actual_token:
        raise credentials_exception
    try:
        payload = jwt.decode(actual_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user is None:
        raise credentials_exception
    return {"username": user["username"], "full_name": user["full_name"], "role": user["role"], "id": user["id"]}

# ----------------- Fast & Layout-Aware OCR Engine -----------------

def preprocess_image_for_ocr(image_bytes: bytes) -> "Image.Image":
    if not CV2_AVAILABLE:
        return Image.open(io.BytesIO(image_bytes))
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return Image.open(io.BytesIO(image_bytes))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1300:
        scale = 1300.0 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 1800:
        scale = 1800.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(thresh)

_paddle_engine = None
_paddle_lock = threading.Lock()

def _get_paddle_engine():
    """Create the PaddleOCR model once with enable_mkldnn=False (safe on Windows Paddle 3.x)."""
    global _paddle_engine
    if _paddle_engine is None:
        with _paddle_lock:
            if _paddle_engine is None:
                try:
                    _paddle_engine = PaddleOCR(
                        lang=PADDLE_LANG,
                        use_doc_orientation_classify=PADDLE_DOC_ORIENTATION,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        enable_mkldnn=False,
                    )
                except TypeError:
                    _paddle_engine = PaddleOCR(lang=PADDLE_LANG, use_angle_cls=False, enable_mkldnn=False)
    return _paddle_engine

def preprocess_image_for_paddle(image_bytes: bytes, enhance_handwritten: bool = False):
    if CV2_AVAILABLE:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1500:
            scale = 1500.0 / max_dim
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        elif w < 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if enhance_handwritten:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return img

    import numpy as _np
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(pil_img.size) > 1500:
        pil_img.thumbnail((1500, 1500))
    return _np.array(pil_img)[:, :, ::-1].copy()

def _paddle_items(results):
    items = []
    for res in results or []:
        if res is None:
            continue
        if hasattr(res, "get") and res.get("rec_texts") is not None:
            texts = res["rec_texts"]
            scores = res.get("rec_scores")
            if scores is None:
                scores = [1.0] * len(texts)
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys")
            for poly, text, score in zip(polys, texts, scores):
                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                items.append((min(xs), min(ys), max(xs), max(ys), text, float(score)))
        elif isinstance(res, list):
            for line in res:
                poly, (text, score) = line[0], line[1]
                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                items.append((min(xs), min(ys), max(xs), max(ys), text, float(score)))
    return items

def _assemble_lines(items) -> str:
    """
    Reconstructs text in TWO ways so both single-column deeds (Timeline2.pdf) and
    3-column boxed records (Testing_realdata1.pdf) parse accurately:
    1) Row-by-row reading order (with ' | ' between horizontal columns).
    2) Vertical Column Cell Pairing: links each box with the value box directly below it
       in the same column as 'HEADER : VALUE'.
    """
    items = [it for it in items if it[4] and it[4].strip()]
    if not items:
        return ""
    heights = sorted(max(1.0, it[3] - it[1]) for it in items)
    med_h = heights[len(heights) // 2] or 12.0
    items_by_y = sorted(items, key=lambda it: (it[1] + it[3]) / 2)

    rows = []
    for it in items_by_y:
        yc = (it[1] + it[3]) / 2
        if rows and abs(yc - rows[-1]["yc"]) <= 0.65 * med_h:
            row = rows[-1]
            row["items"].append(it)
            row["yc"] = (row["yc"] * (len(row["items"]) - 1) + yc) / len(row["items"])
        else:
            rows.append({"yc": yc, "items": [it]})

    lines = []
    for row in rows:
        ordered = sorted(row["items"], key=lambda it: it[0])
        line = ordered[0][4].strip()
        for prev, cur in zip(ordered, ordered[1:]):
            sep = " | " if (cur[0] - prev[2]) > 2.0 * med_h else " "
            line += sep + cur[4].strip()
        lines.append(line)

    vertical_pairs = []
    for i, top in enumerate(items_by_y):
        tx1, ty1, tx2, ty2, t_txt, _ = top
        t_clean = t_txt.strip()
        col_children = []
        cur_bottom_y = ty2
        for bottom in items_by_y[i + 1:]:
            bx1, by1, bx2, by2, b_txt, _ = bottom
            v_gap = by1 - cur_bottom_y
            if v_gap < -0.4 * med_h:
                continue
            if v_gap > 4.2 * med_h:
                break
            overlap = min(tx2, bx2) - max(tx1, bx1)
            min_w = max(1.0, min(tx2 - tx1, bx2 - bx1))
            left_aligned = abs(tx1 - bx1) <= 3.0 * med_h
            if (overlap / min_w >= 0.20 or left_aligned) and (bx1 >= tx1 - 3.5 * med_h):
                col_children.append(b_txt.strip())
                cur_bottom_y = by2
                if len(col_children) >= 3:
                    break
        if col_children:
            vertical_pairs.append(f"{t_clean} : {' '.join(col_children)}")

    if vertical_pairs:
        return "\n".join(lines) + "\n---CELL_PAIRS---\n" + "\n".join(vertical_pairs)
    return "\n".join(lines)

def _run_paddle_on_img(engine, img):
    try:
        results = engine.predict(img)
    except AttributeError:
        results = engine.ocr(img, cls=False)
    return _paddle_items(results)

def _paddle_read(image_bytes: bytes):
    engine = _get_paddle_engine()
    img = preprocess_image_for_paddle(image_bytes, enhance_handwritten=False)
    items = _run_paddle_on_img(engine, img)
    if not items:
        img_enh = preprocess_image_for_paddle(image_bytes, enhance_handwritten=True)
        items = _run_paddle_on_img(engine, img_enh)
        if not items:
            return "", 0.0
    mean_conf = sum(it[5] for it in items) / len(items)
    text = _assemble_lines(items)

    if mean_conf < 0.72 or len(text.strip()) < 100:
        img_enh = preprocess_image_for_paddle(image_bytes, enhance_handwritten=True)
        items_enh = _run_paddle_on_img(engine, img_enh)
        if items_enh:
            conf_enh = sum(it[5] for it in items_enh) / len(items_enh)
            if conf_enh > mean_conf or len(items_enh) > len(items) * 1.15:
                return _assemble_lines(items_enh), conf_enh

    return text, mean_conf

def _ocr_image_bytes(image_bytes: bytes, retry_sparse: bool = True):
    if OCR_ENGINE == "paddle":
        return _paddle_read(image_bytes)
    processed_pil = preprocess_image_for_ocr(image_bytes)
    text = pytesseract.image_to_string(processed_pil, lang=TESSERACT_LANG, config=r"--oem 3 --psm 6")
    if retry_sparse and len(text.strip()) < 50:
        text = pytesseract.image_to_string(processed_pil, lang=TESSERACT_LANG)
    return text, None

def _extract_pdf_structured_text(doc) -> str:
    out_pages = []
    for page_idx in range(min(len(doc), MAX_OCR_PAGES)):
        page = doc[page_idx]
        d = page.get_text("dict") or {}
        line_items = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                bbox = ln.get("bbox", (0, 0, 0, 0))
                txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if txt:
                    line_items.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), txt, 0.99))
        if line_items:
            out_pages.append(_assemble_lines(line_items))
    return "\n".join(out_pages).strip()

def _log_ocr_error(where: str):
    print(f"[OCR] {OCR_ENGINE} failed during {where}:")
    traceback.print_exc()

def run_ocr(file_content: bytes, filename: str) -> Tuple[str, float]:
    file_hash = hashlib.sha256(file_content).hexdigest()
    if file_hash in _OCR_CACHE and _OCR_CACHE[file_hash][0]:
        return _OCR_CACHE[file_hash]

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    text = ""
    quality = 0.90

    if ext == "txt":
        res = (file_content.decode("utf-8", errors="ignore").strip(), 1.0)
        _OCR_CACHE[file_hash] = res
        return res

    if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"):
        if OCR_AVAILABLE:
            try:
                text, conf = _ocr_image_bytes(file_content)
                if len(text.strip()) < 50:
                    quality = 0.55
                elif conf is not None:
                    quality = round(min(0.99, conf), 2)
                else:
                    quality = 0.90
            except Exception:
                _log_ocr_error("image upload")
                text = ""
                quality = 0.0
        else:
            quality = 0.0

    elif ext == "pdf":
        page_pngs = []
        if FITZ_AVAILABLE:
            try:
                doc = fitz.open(stream=file_content, filetype="pdf")
                text = _extract_pdf_structured_text(doc)
                if len(text.strip()) < 60 and OCR_AVAILABLE:
                    dpi = 150 if OCR_ENGINE == "paddle" else 220
                    for idx in range(min(len(doc), MAX_OCR_PAGES)):
                        page_pngs.append(doc[idx].get_pixmap(dpi=dpi).tobytes("png"))
                doc.close()
            except Exception:
                _log_ocr_error("PyMuPDF read")

        if len(text.strip()) < 60 and PDF_TEXT_AVAILABLE and not page_pngs:
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_OCR_PAGES])
            except Exception:
                text = ""

        if len(text.strip()) < 60 and not page_pngs and PDF2IMAGE_AVAILABLE and OCR_AVAILABLE:
            try:
                dpi = 150 if OCR_ENGINE == "paddle" else 220
                for page_img in convert_from_bytes(file_content, dpi=dpi, last_page=MAX_OCR_PAGES):
                    buf = io.BytesIO()
                    page_img.save(buf, format="PNG")
                    page_pngs.append(buf.getvalue())
            except Exception:
                _log_ocr_error("pdf2image render")

        if page_pngs and OCR_AVAILABLE:
            ocr_parts = []
            confs = []
            for png in page_pngs:
                try:
                    p_txt, p_conf = _ocr_image_bytes(png, retry_sparse=False)
                    if p_txt.strip():
                        ocr_parts.append(p_txt.strip())
                    if p_conf is not None:
                        confs.append(p_conf)
                except Exception:
                    _log_ocr_error("scanned-PDF page OCR")
            ocr_text = "\n".join(ocr_parts).strip()
            if len(ocr_text) > len(text.strip()):
                text = ocr_text
                quality = round(min(0.96, sum(confs) / len(confs)), 2) if confs else 0.82
        elif text.strip():
            quality = 0.96
        else:
            quality = 0.0

    res = (text.strip(), quality)
    if res[0]:
        _OCR_CACHE[file_hash] = res
    return res

def _warm_up_paddle():
    try:
        _get_paddle_engine()
        print("[OCR] PaddleOCR models loaded and ready.")
    except Exception:
        print("[OCR] PaddleOCR failed to load:")
        traceback.print_exc()

if OCR_ENGINE == "paddle" and OCR_AVAILABLE:
    threading.Thread(target=_warm_up_paddle, daemon=True).start()

# ----------------- Universal Multi-Format Land Identity Parser -----------------

KNOWN_DISTRICTS = [
    "Ajmer", "Alwar", "Anupgarh", "Balotra", "Banswara", "Baran", "Barmer", "Beawar",
    "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Deeg",
    "Dholpur", "Didwana", "Dungarpur", "Ganganagar", "Sri Ganganagar", "Gangapur",
    "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur",
    "Karauli", "Kekri", "Khairthal", "Kota", "Kotputli", "Nagaur", "Neem Ka Thana",
    "Pali", "Phalodi", "Pratapgarh", "Rajsamand", "Salumbar", "Sanchore", "Sawai Madhopur",
    "Shahpura", "Sikar", "Sirohi", "Tonk", "Udaipur", "Chomu", "Amer", "Sanganer"
]

_HOMOGLYPH_MAP = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "И": "U",
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "і": "i", "к": "k", "м": "m",
    "о": "o", "р": "p", "т": "t", "х": "x", "у": "y", "и": "u",
})

_IDENTIFIER_LABEL_WORDS = {
    "no", "no.", "number", "n0", "n0.", "na", "n/a", "-", "nil", "none", "null",
    "num", "id", "code", "details", "particulars", "new", "old", "and", "or",
    "commercial", "agricultural", "residential", "khata", "khasra", "plot", "sector",
    "property", "survey", "gata", "identifiers", "location"
}

def _strip_indic(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\([^\)]*[\u0900-\u097F][^\)]*\)", "", s)
    s = re.sub(r"[\u0900-\u097F]+", "", s)
    s = re.sub(r"\(\s*\)", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()

def _clean_identifier_match(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    val = _strip_indic(raw).translate(_HOMOGLYPH_MAP).strip()
    val = re.sub(r"^(?:No\.?|Number|Num|ID|#)\s*[:\-\.]?\s*", "", val, flags=re.IGNORECASE).strip()
    val = val.strip(":-.,;|()[]{}")
    if not val or val.lower().rstrip(".") in _IDENTIFIER_LABEL_WORDS:
        return None
    if not re.search(r"[0-9]", val):
        return None
    return val

def clean_location_value(raw: Optional[str], is_village: bool = False) -> Optional[str]:
    if not raw:
        return None
    s = _strip_indic(raw).translate(_HOMOGLYPH_MAP).strip()

    # 1. Strip leading compound label remnants ('/ Locality:', '/ Urban Locality:', '1st Location:', 'Village:')
    for _ in range(2):
        s = re.sub(
            r"^(?:[\d]+(?:st|nd|rd|th)?\s*)?(?:\/\s*)?(?:Urban\s+Locality|Revenue\s+Village|Locality|Village|Sector|Location|District|Tehsil|Gram|Mauza)\s*(?:\/\s*(?:Urban\s+)?(?:Locality|Sector|Village))?\s*[:\-\.]*\s*",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

    # 2. Cut off at any numbered section header ('1. Location', '2. Identifiers') or next-field keyword
    s = re.split(
        r"(?i)(?:\b\d+\.\s*(?:Location|Identifier|Person|Property|Land|Owner|History|Registration|Boundary|Deed)|\b(?:Location\s*(?:Details|Particulars)?|Identifiers?|Person\s*&\s*Holding|Vendee|Vendor|Buyer|Seller|Recorded\s*(?:Holder|Owner)|Claimant|Claimed\s*Share|Share\s*\/?\s*Hissa|Contact|Mobile|Phone|Tehsil|District|Khasra|Khata|Plot\s*No|Total\s*Area)\b)",
        s,
    )[0]

    # 3. Cut off at sentence period
    s = s.split(".")[0]
    s = re.sub(r"[\|\n\r]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-/()")

    if is_village and "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2 and any(parts[-1].lower() == d.lower() for d in KNOWN_DISTRICTS):
            s = parts[0]

    if not s or len(s) < 2 or s.lower() in _IDENTIFIER_LABEL_WORDS:
        return None
    return s.title()

def clean_extracted_name(raw: str) -> str:
    if not raw:
        return ""
    s = _strip_indic(raw).translate(_HOMOGLYPH_MAP).strip()
    s = re.sub(
        r"(?i)^(?:(?:\d+(?:st|nd|rd|th)?\s*)?(?:Recorded\s*(?:Holder|Owner)|Legal\s*Heir|Owner|Vendor|Vendee|Seller|Buyer|Prior\s*Transferor|Transferor|Claimant)\s*(?:\([^\)]*\))?\s*[:\-]?\s*)+",
        "",
        s,
    ).strip()
    s = re.sub(r"(?i)^(?:late\s+)?(?:shri|smt\.?|mr\.?|mrs\.?|ms\.?|dr\.?|kumari)\s+", "", s)
    s = re.sub(
        r"(?i)\b(?:s/o|d/o|w/o|c/o|aged|age\b|r/o|resident\s+of|residing\s+at|son\s+of|daughter\s+of|wife\s+of|widow\s+of|\(deceased\)|deceased|legal\s+heir|caste|occupation|contact|mobile|claimed\s+share|share|hissa|hereinafter)\b.*",
        "",
        s,
    )
    s = re.sub(r"[\.,;:_~\(\)\[\]\|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    words = []
    for w in s.split():
        if w.upper() in ("LLP", "PVT", "LTD", "INC", "JDA", "RIICO", "HUF", "CO") or "&" in w:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)

def normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.translate(_HOMOGLYPH_MAP).strip()
    s = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"[,\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for fmt in (
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y",
        "%b %d %Y", "%B %d %Y", "%d-%m-%y", "%d/%m/%y"
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if 1900 <= dt.year <= 2100:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def _parse_indian_amount(raw: str) -> Optional[float]:
    if not raw:
        return None
    s = raw.strip()
    if re.search(r"\b0\s*\/\-|\b0\.00\b", s):
        return 0.0
    mult = 1.0
    if re.search(r"\b(?:crore|crores|cr\.?)\b", s, re.IGNORECASE):
        mult = 10000000.0
    elif re.search(r"\b(?:lakh|lakhs|lac|lacs)\b", s, re.IGNORECASE):
        mult = 100000.0
    num_m = re.search(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", s)
    if not num_m:
        return None
    cleaned = num_m.group(1).replace(",", "")
    try:
        return round(float(cleaned) * mult, 2)
    except ValueError:
        return None

def extract_land_identity(text: str):
    result = {
        "document_type": "Sale Deed",
        "serial_number": None,
        "district": None,
        "tehsil": None,
        "village": None,
        "khasra_number": None,
        "khata_number": None,
        "plot_number": None,
        "property_id": None,
        "survey_number": None,
        "owner_name": None,
        "seller_name": None,
        "buyer_name": None,
        "share_hissa": "1/1 (Full)",
        "total_area": None,
        "sub_plot_areas": [],
        "land_use": "Agricultural",
        "boundaries_desc": None,
        "consideration_amount": None,
        "registration_date": None,
        "mutation_date": None,
    }
    confidence: Dict[str, float] = {}
    if not text:
        return result, confidence

    t_raw = text.translate(_HOMOGLYPH_MAP).replace("$", "")
    t = _strip_indic(t_raw)
    upper = t.upper()

    # Document Type
    if "SUCCESSION" in upper or "FAUTI" in upper or "INHERITANCE" in upper:
        result["document_type"] = "Succession / Mutation"
    elif "KHATAUNI" in upper or "JAMABANDI" in upper:
        result["document_type"] = "Khasra-Khatauni Record"
    elif "RECORD OF RIGHTS" in upper or re.search(r"\bROR\b", upper):
        result["document_type"] = "Record of Rights (RoR)"
    elif "PATTA" in upper or "LEASE" in upper:
        result["document_type"] = "Patta / Lease"
    elif "CONVEYANCE" in upper or "SALE DEED" in upper or "DEED #" in upper:
        result["document_type"] = "Sale Deed"

    # 1. Serial / Deed Number
    serial_patterns = [
        r"(?:Deed|Serial|Stamp|e-Stamp|Document|Certificate)\s*(?:No\.?|N0\.?|Number|#)\s*[:\.\-]?\s*([A-Z0-9][A-Z0-9\-\/]{3,24})",
        r"\bDeed\s*#\s*([A-Z0-9\-\/]{4,24})",
        r"\b((?:MUT|RJ|IN-RJ|SD|DEED)[\-][A-Z0-9\-]{4,24})\b",
    ]
    for sp in serial_patterns:
        for m in re.finditer(sp, t, re.IGNORECASE):
            cand = m.group(1).strip(" .,-:")
            if re.search(r"[0-9]", cand) and cand.lower() not in _IDENTIFIER_LABEL_WORDS:
                result["serial_number"] = cand
                confidence["serial_number"] = 0.96
                break
        if result["serial_number"]:
            break

    # 2. Registration Date & Mutation Date
    date_tok = r"([0-9]{1,2}[\-\/\s][A-Za-z0-9]{2,9}[\-\/\s][0-9]{2,4})"
    reg_patterns = [
        r"Registration\s*Date\s*[:\-]?\s*" + date_tok,
        r"(?:SRO[^\n\r]*?|Deed[^\n\r]*?)\bDated\s*[:\-]?\s*" + date_tok,
        r"\bDated\s*[:\-]?\s*" + date_tok,
        r"\bDATE\s*[:\-]?\s*" + date_tok,
        r"executed\s+on\s*[:\-]?\s*" + date_tok,
    ]
    for rp in reg_patterns:
        for m in re.finditer(rp, t, re.IGNORECASE):
            d_val = normalize_date(m.group(1))
            if d_val:
                result["registration_date"] = d_val
                confidence["registration_date"] = 0.96
                break
        if result["registration_date"]:
            break

    mut_patterns = [
        r"Mutation\s*Date\s*[:\-]?\s*" + date_tok,
        r"(?:Mutation[^\n\r]*?)\bOrder\s*Date\s*[:\-]?\s*" + date_tok,
        r"\bOrder\s*Date\s*[:\-]?\s*" + date_tok,
    ]
    for mp in mut_patterns:
        for m in re.finditer(mp, t, re.IGNORECASE):
            d_val = normalize_date(m.group(1))
            if d_val:
                result["mutation_date"] = d_val
                confidence["mutation_date"] = 0.92
                break
        if result["mutation_date"]:
            break

    # 3. Location: District, Tehsil, Village / Locality
    m_dist = re.search(r"\b(?:District|Distt\.?|Zila)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,20}?)(?:,|\n|\||\.|Tehsil|Village|$)", t, re.IGNORECASE)
    if m_dist:
        d_clean = clean_location_value(m_dist.group(1), is_village=False)
        if d_clean:
            result["district"] = d_clean
            confidence["district"] = 0.98
    if not result["district"]:
        for kd in KNOWN_DISTRICTS:
            if re.search(rf"\b{re.escape(kd)}\b", t, re.IGNORECASE):
                result["district"] = kd
                confidence["district"] = 0.94
                break

    m_teh = re.search(r"\b(?:Tehsil|Tahsil|Taluka)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,20}?)(?:,|\n|\||\.|Village|Urban|Locality|Sector|District|$)", t, re.IGNORECASE)
    if m_teh:
        t_clean = clean_location_value(m_teh.group(1), is_village=False)
        if t_clean:
            result["tehsil"] = t_clean
            confidence["tehsil"] = 0.96
    elif result["district"]:
        result["tehsil"] = result["district"]
        confidence["tehsil"] = 0.85

    # Village / Urban Locality / Sector (NO variable-length lookbehind; checks prefix in Python!)
    vill_patterns = [
        r"\b(?:Village\s*\/\s*(?:Urban\s*)?(?:Locality|Sector)|Urban\s*Locality|Revenue\s*Village|Village|Locality|Gram|Mauza)\s*[:\-]\s*([^\n\r\|]+(?:\n(?![A-Z\s]{4,}(?:NUMBER|NO|RECORD|IDENTIFIER|PERSON|LOCATION))[^\n\r\|]+)?)",
        r"(?:situated|located|lying)\s+(?:at|in)\s+(?:Village|Gram|Mauza|Locality)?\s*([A-Za-z0-9][A-Za-z0-9\s\-]{2,30}?)(?:,|\.|Tehsil|District|\n)",
        r"\bR\/O\s+Village\s+([A-Za-z][A-Za-z\s]{2,25}?)(?:,|\.|\n|Vendee|Vendor|$)",
    ]
    for idx_vp, vp in enumerate(vill_patterns):
        for m_v in re.finditer(vp, t, re.IGNORECASE):
            # Ignore 'R/O Village ...' or 'Resident of Village ...' on the primary pattern
            if idx_vp == 0:
                prefix = t[max(0, m_v.start() - 18):m_v.start()].lower()
                if "r/o" in prefix or "resident of" in prefix:
                    continue
            v_clean = clean_location_value(m_v.group(1), is_village=True)
            if v_clean:
                result["village"] = v_clean
                confidence["village"] = 0.95
                break
        if result["village"]:
            break

    # 4. Land Identifiers: Khata, Khasra, Plot, Property ID
    khata_patterns = [
        r"KHATA\s*NUMBER\s*[:\-]?\s*(?:\n\s*)?(?:Commercial\s+Khata\s+|Agricultural\s+Khata\s+|Khata\s+)?([A-Z0-9][A-Z0-9\/\-]*)",
        r"\bKhata\s*(?:No\.?|Number)?\s*[:\-]\s*(?:Commercial\s+Khata\s+|Khata\s+)?([A-Z0-9][A-Z0-9\/\-]*)",
        r"\b(?:Commercial|Agricultural)\s+Khata\s+([0-9][A-Z0-9\/\-]*)",
    ]
    for kp in khata_patterns:
        for m in re.finditer(kp, t, re.IGNORECASE):
            val = _clean_identifier_match(m.group(1))
            if val:
                result["khata_number"] = val
                confidence["khata_number"] = 0.93
                break
        if result["khata_number"]:
            break

    khasra_patterns = [
        r"KHASRA\s*NUMBER\s*[:\-]?\s*(?:\n\s*)?(?:Khasra\s+)?([0-9][0-9\/\-A-Za-z]*)",
        r"\b(?:Khasra|Survey|Gata|Killa)\s*(?:No\.?|Number)?\s*[:\-]\s*(?:Khasra\s+)?([0-9][0-9\/\-A-Za-z]*)",
        r"\bKhasra\s+([0-9][0-9\/\-A-Za-z]*)",
    ]
    for kp in khasra_patterns:
        for m in re.finditer(kp, t, re.IGNORECASE):
            val = _clean_identifier_match(m.group(1))
            if val:
                result["khasra_number"] = val
                result["survey_number"] = val
                confidence["khasra_number"] = 0.95
                break
        if result["khasra_number"]:
            break

    plot_patterns = [
        r"\bPlot\s*(?:No\.?|Number)\s*[:\-\.]?\s*([A-Za-z0-9][A-Za-z0-9\-\/]*(?:,\s*Sector\s*[A-Za-z0-9]+)?)",
        r"PLOT\s*NO\.\s*&\s*PROPERTY\s*ID\s*[:\-]?\s*(?:\n\s*)?(?:Plot\s*No\.?\s*)?([A-Za-z0-9][A-Za-z0-9\-\/]*(?:,\s*Sector\s*[A-Za-z0-9]+)?)",
    ]
    for pp in plot_patterns:
        for m in re.finditer(pp, t, re.IGNORECASE):
            val = _clean_identifier_match(m.group(1))
            if val:
                result["plot_number"] = val
                confidence["plot_number"] = 0.92
                break
        if result["plot_number"]:
            break

    prop_patterns = [
        r"\b((?:UPID|ULPIN|PID|PROP)\-[A-Z0-9\-_\/]{3,28})\b",
        r"Property\s*ID\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9\-_\/]{2,24})",
    ]
    for pp in prop_patterns:
        for m in re.finditer(pp, t, re.IGNORECASE):
            val = _clean_identifier_match(m.group(1))
            if val:
                result["property_id"] = val
                confidence["property_id"] = 0.94
                break
        if result["property_id"]:
            break

    # 5. Person / Holding: Owner/Buyer, Seller, Share/Hissa
    owner_patterns = [
        r"RECORDED\s*HOLDER\s*(?:\/\s*OWNER)?\s*(?:\([^\)]*\))?\s*[:\-]?\s*(?:\n\s*)?([A-Za-z][A-Za-z0-9\s\.\&\-]{2,55}?)(?:,|\n|\||SELLER|VENDOR|S\/O|D\/O|W\/O|Son\s+of|$)",
        r"(?:VENDEE|BUYER|PURCHASER|ALLOTTEE|CLAIMANT\s*\/\s*RECORDED\s*OWNER)\s*(?:\([^\)]*\))?\s*[:\-]\s*([A-Za-z][A-Za-z0-9\s\.\&\-]{2,55}?)(?:,|\n|\||S\/O|D\/O|W\/O|Son\s+of|$)",
    ]
    for op in owner_patterns:
        for m in re.finditer(op, t, re.IGNORECASE):
            cleaned = clean_extracted_name(m.group(1))
            if len(cleaned) >= 3 and cleaned.lower() not in _IDENTIFIER_LABEL_WORDS:
                result["buyer_name"] = cleaned
                confidence["buyer_name"] = 0.94
                break
        if result["buyer_name"]:
            break

    seller_patterns = [
        r"(?:SELLER\s*\/\s*PRIOR\s*TRANSFEROR|VENDOR|SELLER|PRIOR\s*TRANSFEROR|TRANSFEROR)\s*(?:\([^\)]*\))?\s*[:\-]?\s*(?:\n\s*)?([A-Za-z][A-Za-z0-9\s\.\&\-]{2,55}?)(?:,|\n|\||SHARE|HISSA|RECORDED|S\/O|D\/O|W\/O|Son\s+of|$)",
    ]
    for sp in seller_patterns:
        for m in re.finditer(sp, t, re.IGNORECASE):
            cleaned = clean_extracted_name(m.group(1))
            if len(cleaned) >= 3 and cleaned.lower() not in _IDENTIFIER_LABEL_WORDS:
                result["seller_name"] = cleaned
                confidence["seller_name"] = 0.93
                break
        if result["seller_name"]:
            break

    result["owner_name"] = result["buyer_name"] or result["seller_name"]
    if result["owner_name"]:
        confidence["owner_name"] = confidence.get("buyer_name", confidence.get("seller_name", 0.92))

    m_sh = re.search(r"(?:SHARE\s*\/?\s*HISSA|Claimed\s*Share)\s*[:\-]?\s*(?:\n\s*)?([0-9]+\/[0-9]+(?:\s*\([^\)]+\))?)", t, re.IGNORECASE)
    if m_sh:
        result["share_hissa"] = m_sh.group(1).strip()
        confidence["share_hissa"] = 0.96

    # 6. Total Area
    m_bigha = re.search(r"(?:Total\s*Area|Area)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*bigha", t, re.IGNORECASE)
    if m_bigha:
        result["total_area"] = float(m_bigha.group(1))
        confidence["total_area"] = 0.96
    else:
        m_sqyd = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Sq\.?\s*Yards?", t, re.IGNORECASE)
        m_ha = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Ha\b", t, re.IGNORECASE)
        if m_sqyd:
            result["total_area"] = float(m_sqyd.group(1))
            confidence["total_area"] = 0.94
        elif m_ha:
            result["total_area"] = float(m_ha.group(1))
            confidence["total_area"] = 0.93

    sub_matches = re.findall(r"Sub-plot\s*[A-Za-z0-9]*\s*=\s*([0-9\.]+)", t, re.IGNORECASE)
    if sub_matches:
        result["sub_plot_areas"] = [float(x) for x in sub_matches]

    # 7. Land Use
    m_lu = re.search(r"Land\s*Use\s*[:\-]\s*([A-Za-z][A-Za-z\s\(\)\/]+?)(?:,|\n|\||\.|Consideration|Total|$)", t, re.IGNORECASE)
    if m_lu:
        result["land_use"] = m_lu.group(1).strip().title()
        confidence["land_use"] = 0.94
    elif "COMMERCIAL" in upper:
        result["land_use"] = "Commercial (Converted)"
        confidence["land_use"] = 0.92
    elif "RESIDENTIAL" in upper or "ABADI" in upper:
        result["land_use"] = "Residential"
        confidence["land_use"] = 0.91

    # 8. Consideration Value
    cons_patterns = [
        r"CONSIDERATION\s*(?:VALUE|AMOUNT)?\s*[:\-]?\s*(?:\n\s*)?(?:Rs\.?|INR|₹)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?|0\/\-)",
        r"(?:Sale\s*Price|Sale\s*Value|Consideration)\s*[:\-]\s*(?:Rs\.?|INR|₹)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    ]
    for cp in cons_patterns:
        for m_c in re.finditer(cp, t, re.IGNORECASE):
            amt = _parse_indian_amount(m_c.group(1))
            if amt is not None:
                result["consideration_amount"] = amt
                confidence["consideration_amount"] = 0.98
                break
        if result["consideration_amount"] is not None:
            break

    # 9. Boundaries / Chauhaddi
    dir_parts = []
    for dir_name in ("North", "South", "East", "West"):
        dm = re.search(
            rf"\b{dir_name}\b\s*(?:\([^\)]*\))?\s*[:\-–]\s*([^\|\n\r]+(?:\n(?!\s*(?:North|South|East|West|\-\-\-))[^\|\n\r]+)?)",
            t,
            re.IGNORECASE,
        )
        if dm:
            val_d = re.sub(r"\s+", " ", dm.group(1)).strip(" ,;.|")
            val_d = re.sub(r"\s*\-\-\-CELL_PAIRS\-\-\-.*$", "", val_d).strip()
            if val_d:
                dir_parts.append(f"{dir_name}: {val_d}")
    if len(dir_parts) >= 2:
        result["boundaries_desc"] = " | ".join(dir_parts)
        confidence["boundaries_desc"] = 0.94
    else:
        m_b = re.search(r"(?:GEOSPATIAL\s*)?BOUNDARIES[^\n]*[:\-]\s*([^\n\r]+)", t, re.IGNORECASE)
        if m_b:
            result["boundaries_desc"] = m_b.group(1).strip()
            confidence["boundaries_desc"] = 0.90

    return result, confidence

def run_validation_rules(data: dict, conn: sqlite3.Connection, current_record_id: Optional[int] = None) -> List[str]:
    flags = []

    if not data.get("khasra_number") and not data.get("plot_number"):
        flags.append("Incomplete Extraction: Missing Khasra / Plot Identifier")
    if not data.get("owner_name"):
        flags.append("Incomplete Extraction: Missing Recorded Owner / Holder Name")

    reg_date_str = data.get("registration_date")
    mut_date_str = data.get("mutation_date")
    if reg_date_str and mut_date_str:
        try:
            reg_d = datetime.strptime(reg_date_str, "%Y-%m-%d")
            mut_d = datetime.strptime(mut_date_str, "%Y-%m-%d")
            if mut_d < reg_d:
                flags.append(f"Timeline Anomaly: Mutation date ({mut_date_str}) precedes registration date ({reg_date_str})")
        except ValueError:
            pass

    total_area = data.get("total_area")
    sub_plots = data.get("sub_plot_areas") or []
    if total_area is not None and len(sub_plots) > 0:
        sub_sum = round(sum(sub_plots), 2)
        if round(float(total_area), 2) != sub_sum:
            flags.append(f"Area Sum Mismatch: Total area ({total_area} Bigha) != sum of sub-plots ({sub_sum} Bigha)")

    if not data.get("district"):
        flags.append("Missing Mandatory Location Identifier: 'District' is unassigned")

    serial = data.get("serial_number")
    if serial:
        query = "SELECT id FROM records WHERE serial_number = ?"
        params = [serial]
        if current_record_id:
            query += " AND id != ?"
            params.append(current_record_id)
        dup = conn.execute(query, params).fetchall()
        if dup:
            flags.append(f"Duplicate Serial Warning: Serial '{serial}' already registered under record #{dup[0]['id']}")

    khasra = data.get("khasra_number")
    village = data.get("village")
    district = data.get("district")
    seller = (data.get("seller_name") or "").strip().lower()
    reg_date = data.get("registration_date")
    if khasra and village and district and seller:
        query = """
            SELECT id, buyer_name, registration_date FROM records
            WHERE LOWER(TRIM(khasra_number)) = LOWER(TRIM(?))
              AND LOWER(TRIM(village)) = LOWER(TRIM(?))
              AND LOWER(TRIM(district)) = LOWER(TRIM(?))
        """
        params = [khasra, village, district]
        if current_record_id:
            query += " AND id != ?"
            params.append(current_record_id)
        prior_records = conn.execute(query, params).fetchall()

        candidates = []
        for pr in prior_records:
            pr_date = pr["registration_date"]
            if reg_date and pr_date:
                try:
                    if datetime.strptime(pr_date, "%Y-%m-%d") < datetime.strptime(reg_date, "%Y-%m-%d"):
                        candidates.append(pr)
                except ValueError:
                    candidates.append(pr)
            elif not reg_date:
                candidates.append(pr)

        if candidates:
            candidates.sort(key=lambda p: p["registration_date"] or "", reverse=True)
            latest_prior = candidates[0]
            prior_buyer = (latest_prior["buyer_name"] or "").strip().lower()
            if prior_buyer and prior_buyer != seller:
                flags.append(
                    f"Chain of Title Break: Seller '{data.get('seller_name')}' does not match the "
                    f"previously recorded owner '{latest_prior['buyer_name']}' (Record #{latest_prior['id']})"
                )

    return flags

def repair_existing_records_in_db():
    """Cleans up existing records in `terra_digitize.db` on startup."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM records").fetchall()
        for r in rows:
            rec_id = r["id"]
            raw_text = r["raw_text"] or ""
            file_path = r["file_path"] or ""

            if (len(raw_text.strip()) < 60 or (not r["khasra_number"] and not r["owner_name"])) and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        raw_text, _ = run_ocr(f.read(), r["filename"] or file_path)
                except Exception:
                    pass

            if not raw_text.strip():
                continue

            extracted, conf = extract_land_identity(raw_text)
            new_village = extracted["village"] or clean_location_value(r["village"], is_village=True)
            new_district = extracted["district"] or clean_location_value(r["district"], is_village=False)
            new_tehsil = extracted["tehsil"] or clean_location_value(r["tehsil"], is_village=False)
            new_owner = extracted["owner_name"] or clean_extracted_name(r["owner_name"] or "")
            new_seller = extracted["seller_name"] or clean_extracted_name(r["seller_name"] or "")
            new_buyer = extracted["buyer_name"] or clean_extracted_name(r["buyer_name"] or "")

            merged = {
                "document_type": extracted["document_type"] or r["document_type"],
                "serial_number": extracted["serial_number"] or r["serial_number"],
                "district": new_district,
                "tehsil": new_tehsil,
                "village": new_village,
                "khasra_number": extracted["khasra_number"] or r["khasra_number"],
                "khata_number": extracted["khata_number"] or r["khata_number"],
                "plot_number": extracted["plot_number"] or r["plot_number"],
                "property_id": extracted["property_id"] or r["property_id"],
                "survey_number": extracted["survey_number"] or r["survey_number"],
                "owner_name": new_owner or None,
                "seller_name": new_seller or None,
                "buyer_name": new_buyer or None,
                "share_hissa": extracted["share_hissa"] or r["share_hissa"],
                "total_area": extracted["total_area"] if extracted["total_area"] is not None else r["total_area"],
                "sub_plot_areas": extracted["sub_plot_areas"],
                "land_use": extracted["land_use"] or r["land_use"],
                "boundaries_desc": extracted["boundaries_desc"] or r["boundaries_desc"],
                "consideration_amount": extracted["consideration_amount"] if extracted["consideration_amount"] is not None else r["consideration_amount"],
                "registration_date": extracted["registration_date"] or r["registration_date"],
                "mutation_date": extracted["mutation_date"] or r["mutation_date"],
            }
            flags = run_validation_rules(merged, conn, current_record_id=rec_id)
            cur_status = r["status"]
            if cur_status != PUBLIC_AWAITING_STATUS:
                cur_status = "Verified" if len(flags) == 0 else "Flagged"

            conn.execute("""
                UPDATE records SET
                    raw_text = ?, district = ?, tehsil = ?, village = ?,
                    khasra_number = ?, khata_number = ?, plot_number = ?, property_id = ?,
                    owner_name = ?, seller_name = ?, buyer_name = ?, share_hissa = ?,
                    total_area = ?, land_use = ?, boundaries_desc = ?, consideration_amount = ?,
                    registration_date = ?, mutation_date = ?, serial_number = ?,
                    confidence_scores = ?, validation_flags = ?, status = ?
                WHERE id = ?
            """, (
                raw_text, merged["district"], merged["tehsil"], merged["village"],
                merged["khasra_number"], merged["khata_number"], merged["plot_number"], merged["property_id"],
                merged["owner_name"], merged["seller_name"], merged["buyer_name"], merged["share_hissa"],
                merged["total_area"], merged["land_use"], merged["boundaries_desc"], merged["consideration_amount"],
                merged["registration_date"], merged["mutation_date"], merged["serial_number"],
                json.dumps(conf), json.dumps(flags), cur_status, rec_id
            ))
        conn.commit()
        conn.close()
        print("[DB] Existing records cleaned and repaired successfully.")
    except Exception:
        traceback.print_exc()

threading.Thread(target=repair_existing_records_in_db, daemon=True).start()

# ----------------- Core API Endpoints -----------------

def find_conflicting_records(conn: sqlite3.Connection, khasra, village, district, exclude_id: Optional[int] = None):
    if not (khasra and village and district):
        return []
    query = """
        SELECT id, submitted_by_user_id, submission_source, status FROM records
        WHERE LOWER(TRIM(khasra_number)) = LOWER(TRIM(?))
          AND LOWER(TRIM(village)) = LOWER(TRIM(?))
          AND LOWER(TRIM(district)) = LOWER(TRIM(?))
          AND status != 'Verified'
    """
    params = [khasra, village, district]
    if exclude_id:
        query += " AND id != ?"
        params.append(exclude_id)
    return conn.execute(query, params).fetchall()

def _flag_existing_conflict(conn: sqlite3.Connection, record_id: int, message: str):
    row = conn.execute("SELECT validation_flags FROM records WHERE id = ?", (record_id,)).fetchone()
    if not row:
        return
    existing = json.loads(row["validation_flags"]) if row["validation_flags"] else []
    if message not in existing:
        existing.append(message)
    conn.execute("UPDATE records SET has_conflict = 1, validation_flags = ? WHERE id = ?", (json.dumps(existing), record_id))

def _ingest_record(
    conn: sqlite3.Connection,
    file_bytes: bytes,
    filename: str,
    uploader_username: str,
    submission_source: str = "Officer",
    submitted_by_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    file_location = os.path.join(UPLOAD_DIR, filename)
    with open(file_location, "wb") as f:
        f.write(file_bytes)

    raw_text, ocr_quality = run_ocr(file_bytes, filename)
    extracted, field_confidence = extract_land_identity(raw_text)
    flags = run_validation_rules(extracted, conn)

    verification_deadline = None
    has_conflict = 0

    if submission_source == "Public":
        final_status = PUBLIC_AWAITING_STATUS
        verification_deadline = (datetime.utcnow() + timedelta(days=PUBLIC_VERIFICATION_SLA_DAYS)).strftime("%Y-%m-%d")

        conflicts = find_conflicting_records(conn, extracted.get("khasra_number"), extracted.get("village"), extracted.get("district"))
        if conflicts:
            has_conflict = 1
            conflict_ids = ", ".join(f"#{c['id']}" for c in conflicts)
            flags.append(
                f"Competing Claim Detected: This land is also claimed in record(s) {conflict_ids} — "
                f"both submissions must be physically verified before either can be approved."
            )
            for c in conflicts:
                _flag_existing_conflict(
                    conn, c["id"],
                    "Competing Claim Detected: A newer submission also claims this same land — "
                    "both must be physically re-verified before either proceeds."
                )
    else:
        final_status = "Verified" if len(flags) == 0 else "Flagged"

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO records (
            filename, file_path, status, raw_text, owner_name, seller_name, buyer_name,
            share_hissa, document_type, serial_number, survey_number, khasra_number, khata_number,
            plot_number, property_id, total_area, sub_plot_areas, land_use, boundaries_desc,
            village, tehsil, district, consideration_amount, registration_date, mutation_date,
            confidence_scores, validation_flags, uploaded_by, ocr_quality,
            submission_source, submitted_by_user_id, verification_deadline, has_conflict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        file_location,
        final_status,
        raw_text,
        extracted["owner_name"],
        extracted["seller_name"],
        extracted["buyer_name"],
        extracted["share_hissa"],
        extracted["document_type"],
        extracted["serial_number"],
        extracted["survey_number"],
        extracted["khasra_number"],
        extracted["khata_number"],
        extracted["plot_number"],
        extracted["property_id"],
        extracted["total_area"],
        json.dumps(extracted["sub_plot_areas"]),
        extracted["land_use"],
        extracted["boundaries_desc"],
        extracted["village"],
        extracted["tehsil"],
        extracted["district"],
        extracted["consideration_amount"],
        extracted["registration_date"],
        extracted["mutation_date"],
        json.dumps(field_confidence),
        json.dumps(flags),
        uploader_username,
        ocr_quality,
        submission_source,
        submitted_by_user_id,
        verification_deadline,
        has_conflict,
    ))
    record_id = cursor.lastrowid
    conn.commit()

    return {"id": record_id, "filename": filename, "status": final_status, "flags": flags, "has_conflict": bool(has_conflict)}

@app.post("/api/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), conn: sqlite3.Connection = Depends(get_db)):
    user = conn.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    token = create_access_token({"sub": user["username"], "role": user["role"]})
    log_action(conn, user["username"], user["role"], "LOGIN", "User logged in")
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"],
        "full_name": user["full_name"]
    }

class PublicRegisterPayload(BaseModel):
    full_name: str
    username: str
    password: str
    phone_number: Optional[str] = None

@app.post("/api/auth/register-public")
def register_public(payload: PublicRegisterPayload, conn: sqlite3.Connection = Depends(get_db)):
    username = payload.username.strip()
    if not username or not payload.password or not payload.full_name.strip():
        raise HTTPException(status_code=400, detail="Full name, username and password are required")
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="That username is already taken")
    conn.execute(
        "INSERT INTO users (username, full_name, hashed_password, role, phone_number) VALUES (?, ?, ?, ?, ?)",
        (username, payload.full_name.strip(), hash_password(payload.password), "Citizen", (payload.phone_number or "").strip())
    )
    conn.commit()
    log_action(conn, username, "Citizen", "PUBLIC_REGISTER", f"Public citizen self-registered: {payload.full_name.strip()}")
    return {"message": "Account created. You can now sign in and submit your land record for verification."}

@app.post("/api/records/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] == "Citizen":
        raise HTTPException(status_code=403, detail="Citizens must submit land records through the Public Submission portal.")

    processed = []
    for file in files:
        contents = await file.read()
        if not contents:
            continue
        result = _ingest_record(conn, contents, file.filename, current_user["username"], submission_source="Officer")
        log_action(conn, current_user["username"], current_user["role"], "UPLOAD",
                   f"Ingested record #{result['id']} ({file.filename}) - Result: {result['status']}")
        processed.append(result)

    return {"message": f"Processed {len(processed)} document(s)", "records": processed}

@app.post("/api/public/submit")
async def submit_public_record(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] != "Citizen":
        raise HTTPException(status_code=403, detail="Only citizen accounts can use the public submission portal.")

    processed = []
    for file in files:
        contents = await file.read()
        if not contents:
            continue
        result = _ingest_record(
            conn, contents, file.filename, current_user["username"],
            submission_source="Public", submitted_by_user_id=current_user["id"]
        )
        log_action(conn, current_user["username"], current_user["role"], "PUBLIC_SUBMIT",
                   f"Citizen submitted record #{result['id']} ({file.filename}) for physical verification")
        processed.append(result)

    return {"message": f"Submitted {len(processed)} document(s) for physical verification.", "records": processed}

@app.get("/api/public/my-submissions")
def my_public_submissions(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Citizen":
        raise HTTPException(status_code=403, detail="Citizen accounts only.")
    rows = conn.execute("SELECT * FROM records WHERE submitted_by_user_id = ? ORDER BY id DESC", (current_user["id"],)).fetchall()
    result = []
    for r in rows:
        rec = dict(r)
        rec["validation_flags"] = json.loads(rec["validation_flags"]) if rec["validation_flags"] else []
        rec["confidence_scores"] = json.loads(rec["confidence_scores"]) if rec["confidence_scores"] else {}
        rec["verification_checklist"] = json.loads(rec["verification_checklist"]) if rec.get("verification_checklist") else {}
        result.append(rec)
    return result

@app.get("/api/officer/public-queue")
def public_queue(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] not in ["Admin", "Revenue Officer"]:
        raise HTTPException(status_code=403, detail="Access denied.")

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    overdue = conn.execute("""
        SELECT id FROM records
        WHERE submission_source = 'Public' AND status = ?
          AND escalated = 0 AND verification_deadline IS NOT NULL AND verification_deadline < ?
    """, (PUBLIC_AWAITING_STATUS, today_str)).fetchall()
    for row in overdue:
        conn.execute("UPDATE records SET escalated = 1 WHERE id = ?", (row["id"],))
        log_action(conn, "SYSTEM", "System", "ESCALATION",
                   f"Record #{row['id']} missed its physical-verification deadline and was escalated to senior authority review.")
    if overdue:
        conn.commit()

    rows = conn.execute("""
        SELECT r.*, u.full_name AS submitter_name, u.phone_number AS submitter_phone
        FROM records r LEFT JOIN users u ON r.submitted_by_user_id = u.id
        WHERE r.submission_source = 'Public'
        ORDER BY r.escalated DESC, r.verification_deadline ASC, r.id DESC
    """).fetchall()

    result = []
    for r in rows:
        rec = dict(r)
        rec["validation_flags"] = json.loads(rec["validation_flags"]) if rec["validation_flags"] else []
        rec["confidence_scores"] = json.loads(rec["confidence_scores"]) if rec["confidence_scores"] else {}
        rec["verification_checklist"] = json.loads(rec["verification_checklist"]) if rec.get("verification_checklist") else {}
        result.append(rec)
    return result

class VerificationChecklistPayload(BaseModel):
    checks: Dict[str, bool]
    notes: Optional[str] = None

@app.post("/api/officer/verify-checklist/{record_id}")
def verify_checklist(
    record_id: int,
    payload: VerificationChecklistPayload,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] not in ["Admin", "Revenue Officer"]:
        raise HTTPException(status_code=403, detail="Access denied.")
    record = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    checks = payload.checks or {}
    all_confirmed = len(checks) > 0 and all(checks.values())

    existing_flags = json.loads(record["validation_flags"]) if record["validation_flags"] else []
    kept_flags = [f for f in existing_flags if not f.startswith("Competing Claim") and not f.startswith("Physical Verification")]

    if all_confirmed:
        new_status = "Verified"
        verified_by = current_user["full_name"]
        verified_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        kept_flags = []
    else:
        new_status = "Flagged"
        verified_by = None
        verified_at = None
        unchecked = [k for k, v in checks.items() if not v]
        kept_flags.append(
            f"Physical Verification Incomplete: field(s) not confirmed on-site — {', '.join(unchecked) if unchecked else 'none confirmed'}"
        )

    conn.execute("""
        UPDATE records SET
            status = ?, verified_by = ?, verified_at = ?, verification_checklist = ?,
            verification_notes = ?, validation_flags = ?, escalated = 0
        WHERE id = ?
    """, (new_status, verified_by, verified_at, json.dumps(checks), payload.notes or "", json.dumps(kept_flags), record_id))
    conn.commit()

    log_action(conn, current_user["username"], current_user["role"], "PHYSICAL_VERIFICATION",
               f"Record #{record_id} physically verified by {current_user['full_name']} — Result: {new_status}")

    return {"message": "Verification recorded", "status": new_status, "verified_by": verified_by}

@app.get("/api/records")
def list_records(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] == "Citizen":
        raise HTTPException(status_code=403, detail="Citizens cannot access the internal registry. Use /api/public/my-submissions.")
    records = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    role = current_user["role"]

    sanitized = []
    for r in records:
        rec = dict(r)
        rec["validation_flags"] = json.loads(rec["validation_flags"]) if rec["validation_flags"] else []
        rec["confidence_scores"] = json.loads(rec["confidence_scores"]) if rec["confidence_scores"] else {}
        if role == "Clerk":
            rec["owner_name"] = "[RESTRICTED - OFFICER ONLY]"
            rec["seller_name"] = "[RESTRICTED]"
            rec["buyer_name"] = "[RESTRICTED]"
            rec["survey_number"] = "[RESTRICTED]"
            rec["khasra_number"] = "[RESTRICTED]"
            rec["khata_number"] = "[RESTRICTED]"
            rec["serial_number"] = "[RESTRICTED]"
            rec["consideration_amount"] = None
            rec["raw_text"] = "[RESTRICTED]"
        sanitized.append(rec)
    return sanitized

@app.get("/api/timeline/chains")
def get_ownership_chains(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    rows = conn.execute("""
        SELECT id, khasra_number, village, district, seller_name, buyer_name, registration_date, status
        FROM records
        WHERE khasra_number IS NOT NULL AND TRIM(khasra_number) != ''
          AND village IS NOT NULL AND TRIM(village) != ''
        ORDER BY id
    """).fetchall()

    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        rec = dict(r)
        key = (
            (rec["khasra_number"] or "").strip().lower(),
            (rec["village"] or "").strip().lower(),
            (rec["district"] or "").strip().lower(),
        )
        groups.setdefault(key, []).append(rec)

    def sort_key(rec):
        d = rec.get("registration_date")
        if d:
            try:
                return (0, datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                pass
        return (1, datetime.min)

    chains = []
    for key, recs in groups.items():
        if len(recs) < 2:
            continue
        recs_sorted = sorted(recs, key=sort_key)

        hops = []
        has_break = False
        for i, rec in enumerate(recs_sorted):
            continuity_ok = True
            if i > 0:
                prev_buyer = (recs_sorted[i - 1].get("buyer_name") or "").strip().lower()
                cur_seller = (rec.get("seller_name") or "").strip().lower()
                if prev_buyer and cur_seller and prev_buyer != cur_seller:
                    continuity_ok = False
                    has_break = True
            hops.append({
                "record_id": rec["id"],
                "seller": rec.get("seller_name") or "Unknown",
                "buyer": rec.get("buyer_name") or "Unknown",
                "date": rec.get("registration_date"),
                "status": rec.get("status"),
                "continuity_ok": continuity_ok,
            })

        chains.append({
            "khasra_number": recs_sorted[0]["khasra_number"],
            "village": recs_sorted[0]["village"],
            "district": recs_sorted[0]["district"],
            "hops": hops,
            "total_transfers": len(hops),
            "has_break": has_break,
            "last_updated_id": max(r["id"] for r in recs_sorted),
        })

    chains.sort(key=lambda c: c["last_updated_id"], reverse=True)
    chains = chains[:8]

    if current_user["role"] == "Clerk":
        for c in chains:
            for h in c["hops"]:
                h["seller"] = "[RESTRICTED]"
                h["buyer"] = "[RESTRICTED]"

    return chains

@app.get("/api/records/{record_id}/file")
def get_record_file(
    record_id: int,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    record = conn.execute("SELECT file_path, filename, submitted_by_user_id FROM records WHERE id = ?", (record_id,)).fetchone()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    if current_user["role"] == "Citizen" and record["submitted_by_user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="You can only view files from your own submissions.")
    if not os.path.exists(record["file_path"]):
        raise HTTPException(status_code=404, detail="File missing from storage")
    return FileResponse(record["file_path"], filename=record["filename"])

class RecordUpdatePayload(BaseModel):
    district: Optional[str] = None
    tehsil: Optional[str] = None
    village: Optional[str] = None
    khasra_number: Optional[str] = None
    khata_number: Optional[str] = None
    plot_number: Optional[str] = None
    property_id: Optional[str] = None
    owner_name: str
    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None
    share_hissa: Optional[str] = None
    total_area: Optional[float] = None
    land_use: Optional[str] = None
    boundaries_desc: Optional[str] = None
    consideration_amount: Optional[float] = None
    registration_date: Optional[str] = None
    mutation_date: Optional[str] = None
    serial_number: Optional[str] = None
    document_type: Optional[str] = None
    status: str

@app.put("/api/records/{record_id}")
def update_record(
    record_id: int,
    payload: RecordUpdatePayload,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] not in ["Admin", "Revenue Officer"]:
        raise HTTPException(status_code=403, detail="Permission denied")

    flags = run_validation_rules(payload.dict(), conn, current_record_id=record_id)
    final_status = payload.status

    conn.execute("""
        UPDATE records SET
            district = ?, tehsil = ?, village = ?, khasra_number = ?, khata_number = ?,
            plot_number = ?, property_id = ?, owner_name = ?, seller_name = ?, buyer_name = ?,
            share_hissa = ?, total_area = ?, land_use = ?, boundaries_desc = ?,
            consideration_amount = ?, registration_date = ?, mutation_date = ?, serial_number = ?,
            document_type = ?, status = ?, validation_flags = ?
        WHERE id = ?
    """, (
        payload.district,
        payload.tehsil,
        payload.village,
        payload.khasra_number,
        payload.khata_number,
        payload.plot_number,
        payload.property_id,
        payload.owner_name,
        payload.seller_name,
        payload.buyer_name,
        payload.share_hissa,
        payload.total_area,
        payload.land_use,
        payload.boundaries_desc,
        payload.consideration_amount,
        payload.registration_date,
        payload.mutation_date,
        payload.serial_number,
        payload.document_type or "Sale Deed",
        final_status,
        json.dumps([] if final_status == "Verified" else flags),
        record_id
    ))
    conn.commit()
    log_action(conn, current_user["username"], current_user["role"], "VERIFY/UPDATE", f"Record #{record_id} saved as {final_status}")
    return {"message": "Record saved successfully", "status": final_status}

class CreateUserPayload(BaseModel):
    username: str
    full_name: str
    password: str
    role: str

@app.post("/api/users")
def create_new_user(payload: CreateUserPayload, current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied. Admins only.")
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (payload.username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.execute(
        "INSERT INTO users (username, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
        (payload.username.strip(), payload.full_name.strip(), hash_password(payload.password), payload.role)
    )
    conn.commit()
    log_action(conn, current_user["username"], current_user["role"], "USER_CREATE", f"Created user '{payload.username}' ({payload.role})")
    return {"message": "User registered successfully"}

@app.get("/api/users")
def list_users(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    return [dict(u) for u in conn.execute("SELECT id, username, full_name, role, created_at FROM users ORDER BY id DESC").fetchall()]

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    log_action(conn, current_user["username"], current_user["role"], "USER_DELETE", f"Deleted user ID {user_id}")
    return {"message": "User deleted"}

@app.get("/api/audit-logs")
def get_audit_logs(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] == "Admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    return [dict(row) for row in conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 100").fetchall()]

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found next to main.py")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
