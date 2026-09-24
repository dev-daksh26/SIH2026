import os
import re
import io
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

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
    import pytesseract
    from PIL import Image
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
except ImportError:
    OCR_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_TEXT_AVAILABLE = True
except ImportError:
    PDF_TEXT_AVAILABLE = False

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

    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
            ("admin", "System Administrator", hash_password("admin123"), "Admin")
        )
        conn.commit()
    conn.close()

init_db()

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

# ----------------- OCR & Preprocessing -----------------

def preprocess_image_for_ocr(image_bytes: bytes) -> Image.Image:
    if not CV2_AVAILABLE:
        return Image.open(io.BytesIO(image_bytes))
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return Image.open(io.BytesIO(image_bytes))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1500:
        scale = 1500.0 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(thresh)

def run_ocr(file_content: bytes, filename: str) -> (str, float):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    text = ""
    quality = 0.90

    # If uploading text file, read directly without dummy replacement
    if ext == "txt":
        return file_content.decode("utf-8", errors="ignore").strip(), 1.0

    if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif"):
        if OCR_AVAILABLE:
            try:
                processed_pil = preprocess_image_for_ocr(file_content)
                text = pytesseract.image_to_string(processed_pil, lang="eng", config=r"--oem 3 --psm 6")
                if len(text.strip()) < 50:
                    text = pytesseract.image_to_string(processed_pil, lang="eng")
            except Exception:
                text = ""

    elif ext == "pdf":
        if PDF_TEXT_AVAILABLE:
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception:
                text = ""

    return text.strip(), quality

KNOWN_DISTRICTS = ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Alwar", "Bikaner", "Bharatpur", "Sikar", "Chomu", "Amer"]

def clean_extracted_name(raw: str) -> str:
    raw = re.sub(r"(?i)\b(s/o|d/o|w/o|aged|r/o|village|resident of|late|son of|daughter of|wife of)\b.*", "", raw)
    return re.sub(r"[\.,;:_~]", " ", raw).strip().title()

def normalize_date(raw: str) -> Optional[str]:
    raw = raw.strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
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
    t = text or ""
    upper = t.upper()

    if "KHATAUNI" in upper or "KHASRA" in upper:
        result["document_type"] = "Khasra-Khatauni Record"
    elif "RECORD OF RIGHTS" in upper or re.search(r"\bROR\b", upper):
        result["document_type"] = "Record of Rights (RoR)"
    elif "PATTA" in upper:
        result["document_type"] = "Patta / Lease"

    # Serial Number
    m = re.search(r"(?:SERIAL|STAMP)\s*(?:NO|N0|\.)?\s*[:\.\-]?\s*([A-Z0-9\-\/]+)", t, re.IGNORECASE)
    if m:
        result["serial_number"] = m.group(1).strip()
        confidence["serial_number"] = 0.95

    # Registration & Execution Dates
    date_patterns = [
        r"Registration\s*Date\s*[:\-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
        r"executed\s+on\s*[:\-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
        r"DATE\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        r"DATE\s*[:\-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
    ]
    for dp in date_patterns:
        m = re.search(dp, t, re.IGNORECASE)
        if m:
            parsed_d = normalize_date(m.group(1))
            if parsed_d:
                result["registration_date"] = parsed_d
                confidence["registration_date"] = 0.96
                break

    # Mutation Date
    m = re.search(r"Mutation\s*Date\s*[:\-]?\s*([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})", t, re.IGNORECASE)
    if m:
        result["mutation_date"] = normalize_date(m.group(1))
        confidence["mutation_date"] = 0.90

    # Seller & Buyer
    m = re.search(r"(?:VENDOR|SELLER)\s*(?:\(Seller\))?\s*[:\-]?\s*([A-Za-z\s\.]+?)(?:,|\n|S\/O|D\/O|AGED|R\/O|Son of|Wife of)", t, re.IGNORECASE)
    if m:
        result["seller_name"] = clean_extracted_name(m.group(1))
        confidence["seller_name"] = 0.92

    m = re.search(r"(?:VENDEE|BUYER|RECORDED\s*HOLDER|ALLOTTEE)\s*(?:\([^\)]+\))?\s*[:\-]?\s*([A-Za-z\s\.]+?)(?:,|\n|S\/O|D\/O|AGED|R\/O|Son of|Wife of)", t, re.IGNORECASE)
    if m:
        result["buyer_name"] = clean_extracted_name(m.group(1))
        confidence["buyer_name"] = 0.93

    result["owner_name"] = result["buyer_name"] or result["seller_name"]
    confidence["owner_name"] = confidence.get("buyer_name", 0.92)

    # Share / Hissa
    m = re.search(r"(?:Share|Hissa)\s*[:\-]?\s*([0-9\/\sA-Za-z\(\)]+)", t, re.IGNORECASE)
    if m:
        result["share_hissa"] = m.group(1).strip()
        confidence["share_hissa"] = 0.95

    # Location
    m = re.search(r"(?:District)\s*[:\-]?\s*([A-Za-z]+)", t, re.IGNORECASE)
    if m:
        result["district"] = m.group(1).strip().title()
        confidence["district"] = 0.98

    m = re.search(r"(?:Tehsil)\s*[:\-]?\s*([A-Za-z]+)", t, re.IGNORECASE)
    if m:
        result["tehsil"] = m.group(1).strip().title()
        confidence["tehsil"] = 0.95

    m = re.search(r"(?:Village|Locality|Urban Locality)\s*[:\-]?\s*([A-Za-z0-9\s]+?)(?:,|\n|$)", t, re.IGNORECASE)
    if m:
        result["village"] = m.group(1).strip().title()
        confidence["village"] = 0.95

    # Identifiers
    m = re.search(r"Khasra\s*(?:No\.?|Number)?\s*[:\-]?\s*([0-9\/\-A-Za-z]+)", t, re.IGNORECASE)
    if m:
        result["khasra_number"] = m.group(1).strip()
        confidence["khasra_number"] = 0.94

    m = re.search(r"Khata\s*(?:No\.?|Number)?\s*[:\-]?\s*([0-9A-Za-z\/\-]+)", t, re.IGNORECASE)
    if m:
        result["khata_number"] = m.group(1).strip()
        confidence["khata_number"] = 0.91

    m = re.search(r"Plot\s*(?:No\.?|Number)?\s*[:\-]?\s*([0-9A-Za-z\/\-]+)", t, re.IGNORECASE)
    if m:
        result["plot_number"] = m.group(1).strip()
        confidence["plot_number"] = 0.90

    m = re.search(r"Property\s*ID\s*[:\-]?\s*([0-9A-Za-z\-\/_]+)", t, re.IGNORECASE)
    if m:
        result["property_id"] = m.group(1).strip()
        confidence["property_id"] = 0.92

    # Area & Sub-plots
    m = re.search(r"(?:Area|Total\s*Area)\s*[:\-]?\s*([0-9\.]+)\s*(?:bigha|acre|hectare|sq)?", t, re.IGNORECASE)
    if m:
        try:
            result["total_area"] = float(m.group(1))
            confidence["total_area"] = 0.95
        except ValueError:
            pass

    sub_matches = re.findall(r"Sub-plot\s*[A-Za-z0-9]*\s*=\s*([0-9\.]+)", t, re.IGNORECASE)
    if sub_matches:
        result["sub_plot_areas"] = [float(x) for x in sub_matches]

    # Land use
    m = re.search(r"Land\s*Use(?:\s*[\/\(A-Za-z\)]*)?\s*[:\-]?\s*([A-Za-z\s\(\)]+?)(?:,|\n|$)", t, re.IGNORECASE)
    if m:
        result["land_use"] = m.group(1).strip().title()
        confidence["land_use"] = 0.93

    # Boundaries
    m = re.search(r"Boundaries\s*[:\-]?\s*([^\n\r]+)", t, re.IGNORECASE)
    if m:
        result["boundaries_desc"] = m.group(1).strip()
        confidence["boundaries_desc"] = 0.90

    # Consideration amount
    m = re.search(r"(?:Consideration|Amount|Sale\s+Value|Price)\s*[:\-]?\s*(?:Rs\.?|INR)?\s*([0-9,]+(?:\.[0-9]{2})?)", t, re.IGNORECASE)
    if m:
        try:
            val_str = m.group(1).replace(",", "").strip()
            result["consideration_amount"] = float(val_str)
            confidence["consideration_amount"] = 0.98
        except ValueError:
            pass

    return result, confidence

def run_validation_rules(data: dict, conn: sqlite3.Connection, current_record_id: Optional[int] = None) -> List[str]:
    flags = []

    # Rule 1: Chronological Order (Mutation cannot precede Registration)
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

    # Rule 2: Sub-plot Sum Reconciliation
    total_area = data.get("total_area")
    sub_plots = data.get("sub_plot_areas") or []
    if total_area is not None and len(sub_plots) > 0:
        sub_sum = round(sum(sub_plots), 2)
        if round(float(total_area), 2) != sub_sum:
            flags.append(f"Area Sum Mismatch: Total area ({total_area} Bigha) != sum of sub-plots ({sub_sum} Bigha)")

    # Rule 3: Missing Mandatory Location Identifier
    if not data.get("district"):
        flags.append("Missing Mandatory Location Identifier: 'District' is unassigned")

    # Rule 4: Duplicate Stamp Serial Check (colliding only against OTHER records)
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

    return flags

# ----------------- Core API Endpoints -----------------

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

@app.post("/api/records/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    processed = []
    for file in files:
        contents = await file.read()
        if not contents:
            continue

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(contents)

        raw_text, ocr_quality = run_ocr(contents, file.filename)
        extracted, field_confidence = extract_land_identity(raw_text)
        flags = run_validation_rules(extracted, conn)

        # Automatic Classification: Genuine documents with 0 flags are auto-verified
        auto_status = "Verified" if len(flags) == 0 else "Flagged"

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO records (
                filename, file_path, status, raw_text, owner_name, seller_name, buyer_name,
                share_hissa, document_type, serial_number, survey_number, khasra_number, khata_number,
                plot_number, property_id, total_area, sub_plot_areas, land_use, boundaries_desc,
                village, tehsil, district, consideration_amount, registration_date, mutation_date,
                confidence_scores, validation_flags, uploaded_by, ocr_quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file.filename,
            file_location,
            auto_status,
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
            current_user["username"],
            ocr_quality,
        ))
        record_id = cursor.lastrowid
        conn.commit()

        log_action(conn, current_user["username"], current_user["role"], "UPLOAD", f"Ingested record #{record_id} ({file.filename}) - Result: {auto_status}")
        processed.append({"id": record_id, "filename": file.filename, "status": auto_status, "flags": flags})

    return {"message": f"Processed {len(processed)} document(s)", "records": processed}

@app.get("/api/records")
def list_records(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
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

@app.get("/api/records/{record_id}/file")
def get_record_file(
    record_id: int,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    record = conn.execute("SELECT file_path, filename FROM records WHERE id = ?", (record_id)).fetchone()
    if not record or not os.path.exists(record["file_path"]):
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
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    return [dict(row) for row in conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 100").fetchall()]

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html not found next to main.py")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
