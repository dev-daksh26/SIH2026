import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = "bharat-vault-secure-token-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 180
UPLOAD_DIR = "uploads"
DB_FILE = "terra_digitize.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

app = FastAPI(title="BharatVault API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Hashing Helpers -----------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

# ----------------- Database Setup -----------------

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Users Table with full_name
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
    
    # Land Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Uploaded',
        raw_text TEXT,
        owner_name TEXT,
        survey_number TEXT,
        khasra_number TEXT,
        khata_number TEXT,
        total_area REAL,
        sub_plot_areas TEXT,
        village TEXT,
        tehsil TEXT,
        district TEXT,
        registration_date TEXT,
        mutation_date TEXT,
        confidence_scores TEXT,
        validation_flags TEXT,
        uploaded_by TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Audit Trail Table
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
    
    # Seed default initial Admin
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
            ("admin", "System Administrator", hash_password("admin123"), "Admin")
        )
        conn.commit()
    conn.close()

init_db()

# ----------------- Audit Logger Helper -----------------

def log_action(conn: sqlite3.Connection, username: str, role: str, action: str, details: str = ""):
    conn.execute(
        "INSERT INTO audit_logs (user_id, role, action, details) VALUES (?, ?, ?, ?)",
        (username, role, action, details)
    )
    conn.commit()

# ----------------- Auth Helpers -----------------

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), conn: sqlite3.Connection = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session invalid or expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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

# ----------------- Validation Rules Engine -----------------

def run_validation_rules(data: dict, conn: sqlite3.Connection, current_record_id: Optional[int] = None) -> List[str]:
    flags = []
    
    # Area Sum Check
    total_area = data.get("total_area")
    sub_plots = data.get("sub_plot_areas", [])
    if total_area is not None and sub_plots:
        sub_sum = sum(float(x) for x in sub_plots)
        if round(sub_sum, 2) != round(float(total_area), 2):
            flags.append(f"Area Mismatch: Parent area ({total_area}) != sub-plot sum ({sub_sum:.2f})")

    # Chronological Dates Check
    reg_date_str = data.get("registration_date")
    mut_date_str = data.get("mutation_date")
    if reg_date_str and mut_date_str:
        try:
            reg_d = datetime.strptime(reg_date_str, "%Y-%m-%d")
            mut_d = datetime.strptime(mut_date_str, "%Y-%m-%d")
            if mut_d < reg_d:
                flags.append(f"Date Anomaly: Mutation ({mut_date_str}) precedes registration ({reg_date_str})")
        except ValueError:
            flags.append("Invalid Date format: Must be YYYY-MM-DD")

    # Mandatory Fields
    mandatory = ["khasra_number", "village", "district"]
    for field in mandatory:
        if not data.get(field):
            flags.append(f"Missing Field: '{field}' is mandatory.")

    # Duplicate Owner-Plot Conflict
    khasra = data.get("khasra_number")
    village = data.get("village")
    owner = data.get("owner_name")
    if khasra and village:
        query = "SELECT id, owner_name FROM records WHERE khasra_number = ? AND village = ?"
        params = [khasra, village]
        if current_record_id:
            query += " AND id != ?"
            params.append(current_record_id)
        matches = conn.execute(query, params).fetchall()
        for match in matches:
            if owner and match["owner_name"] and match["owner_name"].strip().lower() != owner.strip().lower():
                flags.append(f"Ownership Conflict: Khasra '{khasra}' in '{village}' is already registered to '{match['owner_name']}'")

    return flags

# ----------------- OCR Mock/Structuring -----------------

def parse_simulated_ocr_and_structure(file_content: bytes, filename: str) -> dict:
    text_content = ""
    try:
        text_content = file_content.decode("utf-8")
    except Exception:
        text_content = f"Scanned file: {filename}\nKhasra: 501/2\nOwner: Mohan Das\nArea: 4.0\nVillage: Rampur\nDistrict: Jaipur"

    return {
        "raw_text": text_content,
        "owner_name": "Mohan Das Sharma",
        "survey_number": "SN-902",
        "khasra_number": "501/2",
        "khata_number": "KH-88",
        "total_area": 4.0,
        "sub_plot_areas": [2.0, 1.5],  # Demonstrates sum mismatch (3.5 vs 4.0)
        "village": "Rampur",
        "tehsil": "Amer",
        "district": "Jaipur",
        "registration_date": "2023-04-10",
        "mutation_date": "2022-01-01",  # Demonstrates chronological anomaly
        "confidence_scores": {
            "owner_name": 0.95,
            "khasra_number": 0.89,
            "total_area": 0.70,
            "village": 0.99
        }
    }

# ----------------- Auth & User Management Endpoints -----------------

@app.post("/api/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), conn: sqlite3.Connection = Depends(get_db)):
    user = conn.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    log_action(conn, user["username"], user["role"], "LOGIN", "User authenticated successfully")
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"],
        "full_name": user["full_name"]
    }

class CreateUserPayload(BaseModel):
    username: str
    full_name: str
    password: str
    role: str

@app.post("/api/users")
def create_new_user(
    payload: CreateUserPayload,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied. Only Admins can register new users.")

    if payload.role not in ["Admin", "Revenue Officer", "Clerk"]:
        raise HTTPException(status_code=400, detail="Invalid role specified.")

    existing = conn.execute("SELECT id FROM users WHERE username = ?", (payload.username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username/User ID already exists.")

    hashed = hash_password(payload.password)
    conn.execute(
        "INSERT INTO users (username, full_name, hashed_password, role) VALUES (?, ?, ?, ?)",
        (payload.username.strip(), payload.full_name.strip(), hashed, payload.role)
    )
    conn.commit()

    log_action(conn, current_user["username"], current_user["role"], "USER_CREATE", f"Created user '{payload.username}' with role '{payload.role}'")
    return {"message": f"User {payload.username} created successfully."}

@app.get("/api/users")
def list_users(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Access denied.")
    
    users = conn.execute("SELECT id, username, full_name, role, created_at FROM users ORDER BY id DESC").fetchall()
    return [dict(u) for u in users]

# ----------------- Document & Record Endpoints -----------------

@app.post("/api/records/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(file_location, "wb") as f:
        f.write(contents)

    structured = parse_simulated_ocr_and_structure(contents, file.filename)
    flags = run_validation_rules(structured, conn)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO records (
            filename, file_path, status, raw_text, owner_name, survey_number,
            khasra_number, khata_number, total_area, sub_plot_areas, village,
            tehsil, district, registration_date, mutation_date, confidence_scores,
            validation_flags, uploaded_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file.filename,
        file_location,
        "Pending Verification" if flags else "Ready",
        structured["raw_text"],
        structured["owner_name"],
        structured["survey_number"],
        structured["khasra_number"],
        structured["khata_number"],
        structured["total_area"],
        json.dumps(structured["sub_plot_areas"]),
        structured["village"],
        structured["tehsil"],
        structured["district"],
        structured["registration_date"],
        structured["mutation_date"],
        json.dumps(structured["confidence_scores"]),
        json.dumps(flags),
        current_user["username"]
    ))
    record_id = cursor.lastrowid
    conn.commit()

    log_action(conn, current_user["username"], current_user["role"], "UPLOAD", f"Uploaded record #{record_id} ({file.filename})")
    return {"message": "Document uploaded and parsed", "record_id": record_id, "flags_count": len(flags)}

@app.get("/api/records")
def list_records(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    records = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    role = current_user["role"]

    sanitized = []
    for r in records:
        rec = dict(r)
        rec["sub_plot_areas"] = json.loads(rec["sub_plot_areas"]) if rec["sub_plot_areas"] else []
        rec["confidence_scores"] = json.loads(rec["confidence_scores"]) if rec["confidence_scores"] else {}
        rec["validation_flags"] = json.loads(rec["validation_flags"]) if rec["validation_flags"] else []

        # STRICT RBAC: Redact sensitive fields for Clerk role at API level
        if role == "Clerk":
            rec["owner_name"] = "[RESTRICTED - OFFICER ONLY]"
            rec["survey_number"] = "[RESTRICTED]"
            rec["khasra_number"] = "[RESTRICTED]"
            rec["khata_number"] = "[RESTRICTED]"
            rec["sub_plot_areas"] = []
            rec["raw_text"] = "[RESTRICTED]"
            rec["confidence_scores"] = {}

        sanitized.append(rec)
    return sanitized

class RecordUpdatePayload(BaseModel):
    owner_name: str
    khasra_number: str
    total_area: float
    village: str
    registration_date: str
    mutation_date: str
    status: str

@app.put("/api/records/{record_id}")
def update_and_verify_record(
    record_id: int,
    payload: RecordUpdatePayload,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if current_user["role"] not in ["Admin", "Revenue Officer"]:
        raise HTTPException(status_code=403, detail="Permission denied. Only Revenue Officers or Admins can review records.")

    existing = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Record not found")

    sub_plots = json.loads(existing["sub_plot_areas"]) if existing["sub_plot_areas"] else []
    validation_input = payload.dict()
    validation_input["sub_plot_areas"] = sub_plots
    
    flags = run_validation_rules(validation_input, conn, current_record_id=record_id)
    new_status = payload.status
    if new_status == "Verified" and len(flags) > 0:
        new_status = "Flagged"

    conn.execute("""
        UPDATE records SET
            owner_name = ?, khasra_number = ?, total_area = ?, village = ?,
            registration_date = ?, mutation_date = ?, status = ?, validation_flags = ?
        WHERE id = ?
    """, (
        payload.owner_name,
        payload.khasra_number,
        payload.total_area,
        payload.village,
        payload.registration_date,
        payload.mutation_date,
        new_status,
        json.dumps(flags),
        record_id
    ))
    conn.commit()

    log_action(conn, current_user["username"], current_user["role"], "VERIFY/UPDATE", f"Updated record #{record_id} to status: {new_status}")
    return {"message": "Record updated", "status": new_status, "remaining_flags": flags}

@app.get("/api/dashboard/metrics")
def get_metrics(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    verified = conn.execute("SELECT COUNT(*) FROM records WHERE status = 'Verified'").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM records WHERE status = 'Pending Verification'").fetchone()[0]
    flagged = conn.execute("SELECT COUNT(*) FROM records WHERE status = 'Flagged'").fetchone()[0]

    return {
        "role": current_user["role"],
        "total_documents": total,
        "verified_count": verified,
        "pending_count": pending,
        "flagged_count": flagged
    }

@app.get("/api/audit-logs")
def get_audit_logs(current_user: dict = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if current_user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Only Admins can inspect the audit trail.")
    logs = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(row) for row in logs]

# Serve Frontend SPA
app.mount("/", StaticFiles(directory="static", html=True), name="static")