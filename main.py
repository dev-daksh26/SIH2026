import os
import re
import io
import json
import base64
import urllib.request
import urllib.error
import threading
import traceback
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

# CRITICAL FOR WINDOWS + PADDLE 3.x: prevent oneDNN / PIR C++ crash before importing paddle
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

# ----------------- OCR & AI Normalizer Configuration -----------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").strip().lower()
if OCR_ENGINE not in ("paddle", "tesseract"):
    print(f"[OCR] Unknown OCR_ENGINE '{OCR_ENGINE}' - using 'paddle'")
    OCR_ENGINE = "paddle"

PADDLE_LANG = os.getenv("PADDLE_OCR_LANG", "devanagari")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "hin+eng")
PADDLE_DOC_ORIENTATION = os.getenv("PADDLE_DOC_ORIENTATION", "0") == "1"
PADDLE_ENABLE_MKLDNN = os.getenv("PADDLE_ENABLE_MKLDNN", "0") == "1"
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "6"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

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

print(
    f"[OCR] engine={OCR_ENGINE}  available={OCR_AVAILABLE}  lang={PADDLE_LANG if OCR_ENGINE == 'paddle' else TESSERACT_LANG}  "
    f"normalizer={'Gemini-Vision' if GEMINI_API_KEY else ('Groq-LLM' if GROQ_API_KEY else 'Multilingual-Local-Canonical')}"
    + (f"  error={OCR_INIT_ERROR!r}" if OCR_INIT_ERROR else "")
)

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

app = FastAPI(title="BharatVault Multilingual Land Registry API", version="9.0")
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
        phone_number TEXT,
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
        source_language TEXT DEFAULT 'en',
        owner_name TEXT,
        owner_name_native TEXT,
        seller_name TEXT,
        seller_name_native TEXT,
        buyer_name TEXT,
        buyer_name_native TEXT,
        share_hissa TEXT,
        survey_number TEXT,
        khasra_number TEXT,
        khata_number TEXT,
        plot_number TEXT,
        property_id TEXT,
        total_area REAL,
        sub_plot_areas TEXT,
        land_use TEXT,
        land_use_native TEXT,
        boundaries_desc TEXT,
        boundaries_desc_native TEXT,
        village TEXT,
        village_native TEXT,
        tehsil TEXT,
        tehsil_native TEXT,
        district TEXT,
        district_native TEXT,
        registration_date TEXT,
        mutation_date TEXT,
        serial_number TEXT,
        consideration_amount REAL,
        confidence_scores TEXT,
        validation_flags TEXT,
        document_type TEXT,
        document_type_native TEXT,
        uploaded_by TEXT,
        ocr_quality REAL,
        submission_source TEXT DEFAULT 'Officer',
        submitted_by_user_id INTEGER,
        verification_deadline TEXT,
        escalated INTEGER DEFAULT 0,
        has_conflict INTEGER DEFAULT 0,
        verified_by TEXT,
        verified_at TEXT,
        verification_checklist TEXT,
        verification_notes TEXT,
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
    _safe_add_column("records", "source_language TEXT DEFAULT 'en'")
    _safe_add_column("records", "owner_name_native TEXT")
    _safe_add_column("records", "seller_name_native TEXT")
    _safe_add_column("records", "buyer_name_native TEXT")
    _safe_add_column("records", "village_native TEXT")
    _safe_add_column("records", "tehsil_native TEXT")
    _safe_add_column("records", "district_native TEXT")
    _safe_add_column("records", "land_use_native TEXT")
    _safe_add_column("records", "boundaries_desc_native TEXT")
    _safe_add_column("records", "document_type_native TEXT")
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

# ----------------- MULTILINGUAL & SCRIPT RECONSTRUCTION ENGINE -----------------

# 1. Map Indic/Devanagari numerals to standard digits (०-९ -> 0-9)
_DEVANAGARI_DIGITS = str.maketrans({
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
})

# 2. Map font ligature artifacts and non-standard PDF glyphs to clean Devanagari text
_LIGATURE_REPLACEMENTS = {
    "\u0130": "\u093f",  # İ -> ि
    "\u0244": "न्न",     # Ʉ -> न्न
    "\u0150": "े",       # Ő -> े
    "\u016b": "न्द्र",    # ū -> न्द्र
    "\u012f": "ित",      # į -> ित
    "\u014a": "",        # Ŋ -> empty
    "\u015c": "रु.",     # Ŝ -> रु.
    "\u015d": "रू",      # ŝ -> रू
    "\u0162": "क्र",     # Ţ -> क्र
    "\u0164": "ग्र",     # Ť -> ग्र
    "\u0169": "त्र",     # ũ -> त्र
    "\u016e": "प्र",     # Ů -> प्र
    "\u0175": "श्र",     # ŵ -> श्र
    "\u0183": "क्ट",     # ƃ -> क्ट
    "\u0185": "",        # ƅ -> empty
    "\u0197": "क्ष",     # Ɨ -> क्ष
    "\u01a5": "ख्य",     # ƥ -> ख्य
    "\u0205": "त्त",     # ȅ -> त्त
    "\u0209": "त्नी",    # ȉ -> त्नी
    "\u020f": "त्म",     # ȏ -> त्म
    "\u021e": "द्ध",     # Ȟ -> द्ध
    "\u0258": "प्ल",     # ɘ -> प्ल
    "\u02a9": "व्य",     # ʩ -> व्य
    "\u02ae": "श्चि",    # ʮ -> श्चि
    "\u02b4": "श्या",    # ʴ -> श्या
    "\u02d1": "स्त",     # ˑ -> स्त
    "\u02e2": "स्व",     # ˢ -> स्व
    "\u02e3": "स्स",     # ˣ -> स्स
    "\u033e": "",        # ̾ -> empty
    "रु..": "रु. ",
    "पत्नीी": "पत्नी",
    "श्रीी": "श्री",
    "पहचानकतार्ता": "पहचानकर्ता",
    "पहचानकता": "पहचानकर्ता",
    "व्यकि्ति": "व्यक्ति",
    "संपति्त": "संपत्ति",
    "मागर्ता": "मार्ग",
    "माग,": "मार्ग,",
    "माग": "मार्ग",
    "वमार्ता": "वर्मा",
    "वमा": "वर्मा",
    "पूणर्ता": "पूर्ण",
    "पूण": "पूर्ण",
    "दकि्षेण": "दक्षिण",
    "दकि्षण": "दक्षिण",
    "दिक्षण": "दक्षिण",
    "पूव:": "पूर्व:",
    "पूव": "पूर्व",
    "पशि्चमि:": "पश्चिम:",
    "पशि्चम": "पश्चिम",
    "पिश्चिम": "पश्चिम",
    "चौहद्धी": "चौहद्दी",
    "रूपांततिरत": "रूपांतरित",
    "रूपांतितरत": "रूपांतरित",
    "प्रितफल": "प्रतिफल",
    "प्रतफिल": "प्रतिफल",
    "व्यि": "व्यक्ति",
    "थान:": "स्थान:",
}

_HOMOGLYPH_MAP = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "И": "U",
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "і": "i", "к": "k", "м": "m",
    "о": "o", "р": "p", "т": "t", "х": "x", "у": "y", "и": "u",
})

# 3. Transliteration & Translation Curated Dictionaries
ADMIN_LOCATIONS_MAP = {
    # Districts (Hindi -> English)
    "अजमेर": "Ajmer", "अलवर": "Alwar", "अनूपगढ़": "Anupgarh", "बालोतरा": "Balotra",
    "बांसवाड़ा": "Banswara", "बारां": "Baran", "बाड़मेर": "Barmer", "ब्यावर": "Beawar",
    "भरतपुर": "Bharatpur", "भीलवाड़ा": "Bhilwara", "बीकानेर": "Bikaner", "बूंदी": "Bundi",
    "चित्तौड़गढ़": "Chittorgarh", "चूरू": "Churu", "दौसा": "Dausa", "डीग": "Deeg",
    "धौलपुर": "Dholpur", "डीडवाना": "Didwana", "डूंगरपुर": "Dungarpur", "गंगानगर": "Sri Ganganagar",
    "श्रीगंगानगर": "Sri Ganganagar", "गंगापुर": "Gangapur", "हनुमानगढ़": "Hanumangarh",
    "जयपुर": "Jaipur", "जैसलमेर": "Jaisalmer", "जालोर": "Jalore", "झालावाड़": "Jhalawar",
    "झुंझुनू": "Jhunjhunu", "झुंझुनूं": "Jhunjhunu", "जोधपुर": "Jodhpur", "करौली": "Karauli",
    "केकड़ी": "Kekri", "खैरथल": "Khairthal", "कोटा": "Kota", "कोटपूतली": "Kotputli",
    "नागौर": "Nagaur", "नीमकाथाना": "Neem Ka Thana", "पाली": "Pali", "फलोदी": "Phalodi",
    "प्रतापगढ़": "Pratapgarh", "राजसमंद": "Rajsamand", "सलूम्बर": "Salumbar", "सांचौर": "Sanchore",
    "सवाई माधोपुर": "Sawai Madhopur", "शाहपुरा": "Shahpura", "सीकर": "Sikar", "सिरोही": "Sirohi",
    "टोंक": "Tonk", "उदयपुर": "Udaipur",
    # Notable Villages / Localities
    "किशनगढ़": "Kishangarh", "किशनगढ़": "Kishangarh", "बड़गांव": "Badgaon", "बड़गांव": "Badgaon",
    "लाडपुरा": "Ladpura", "चौमूं": "Chomu", "आमेर": "Amer", "सांगानेर": "Sanganer",
    "मानसरोवर": "Mansarovar", "सालावास": "Salawas", "पाल रोड": "Pal Road", "वैशाली नगर": "Vaishali Nagar"
}

# English -> Hindi reverse lookup
ADMIN_LOCATIONS_REVERSE = {v.lower(): k for k, v in ADMIN_LOCATIONS_MAP.items()}

COMMON_NAMES_MAP = {
    "भागचंद": "Bhagchand", "सुरेन्द्र": "Surendra", "सुरेंद्र": "Surendra", "बन्ना": "Banna",
    "लाल": "Lal", "कुमार": "Kumar", "जाट": "Jat", "मोहन": "Mohan", "सिंह": "Singh",
    "राजपूत": "Rajput", "फतेह": "Fateh", "मंजू": "Manju", "देवी": "Devi", "दिनेश": "Dinesh",
    "जोशी": "Joshi", "ओम": "Om", "प्रकाश": "Prakash", "नागर": "Nagar", "राम": "Ram",
    "बिलास": "Bilas", "विलास": "Vilas", "सुनील": "Sunil", "वर्मा": "Verma",
    "राधेश्याम": "Radheshyam", "शर्मा": "Sharma", "गुप्ता": "Gupta", "मीणा": "Meena",
    "चौधरी": "Choudhary", "यादव": "Yadav", "भंवर": "Bhanwar", "सुनीता": "Sunita"
}

SEMANTIC_CATEGORIES_MAP = {
    # Land Use
    "कृषि": "Agricultural", "कृषि (चाही)": "Agricultural (Chahi / Irrigated)", "चाही": "Agricultural (Chahi)",
    "बारानी": "Agricultural (Barani)", "आवासीय": "Residential", "आवासीय (रूपांतरित)": "Residential (Converted)",
    "व्यावसायिक": "Commercial (Converted)", "वाणिज्यिक": "Commercial (Converted)", "औद्योगिक": "Industrial",
    # Document Type
    "विक्रय विलेख": "Sale Deed", "विक्रय विलेख (बैनामा)": "Sale Deed", "बैनामा": "Sale Deed",
    "हस्तांतरण विलेख": "Transfer Deed", "नामांतरण": "Succession / Mutation", "फौती नामांतरण": "Succession / Mutation",
    "जमाबंदी": "Khasra-Khatauni Record", "खतौनी": "Khasra-Khatauni Record", "पट्टा": "Patta / Lease",
    "पट्टा / लीज": "Patta / Lease",
    # Share / Hissa
    "1/1 (पूर्ण)": "1/1 (Full)", "पूर्ण": "1/1 (Full)", "1/2 भाग": "1/2 Share", "1/4 भाग": "1/4 Share",
    # Direction
    "उत्तर": "North", "दक्षिण": "South", "पूर्व": "East", "पश्चिम": "West",
}

# Phonetic Devanagari Transliterator for unlisted Indian words
_VOWEL_SIGNS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
    "ं": "n", "ँ": "n", "ः": "h"
}
_INDEPENDENT_VOWELS = {
    "अ": "A", "आ": "Aa", "इ": "I", "ई": "I", "उ": "U", "ऊ": "U",
    "ए": "E", "ऐ": "Ai", "ओ": "O", "औ": "Au", "ऋ": "Ri"
}
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh",
    "ग़": "gh", "ज़": "z", "फ़": "f", "ड़": "d", "ढ़": "dh"
}

def transliterate_indic_to_english(text: str) -> str:
    """Phonetic transliteration of Indian names and locations with curated dictionary lookup."""
    if not text:
        return ""
    words = text.strip().split()
    output_words = []
    for w in words:
        clean_w = w.strip(",;.:/()-")
        if clean_w in ADMIN_LOCATIONS_MAP:
            output_words.append(ADMIN_LOCATIONS_MAP[clean_w])
            continue
        if clean_w in COMMON_NAMES_MAP:
            output_words.append(COMMON_NAMES_MAP[clean_w])
            continue
        # Generic character-by-character transliteration
        chars = list(clean_w)
        res = []
        i = 0
        n = len(chars)
        while i < n:
            c = chars[i]
            if i + 1 < n and chars[i + 1] == "़":
                combined = c + "़"
                if combined in _CONSONANTS:
                    c = combined
                    i += 1
            if c in _INDEPENDENT_VOWELS:
                res.append(_INDEPENDENT_VOWELS[c])
            elif c in _CONSONANTS:
                base = _CONSONANTS[c]
                if i + 1 < n and chars[i + 1] == "्":
                    res.append(base)
                    i += 1
                elif i + 1 < n and chars[i + 1] in _VOWEL_SIGNS:
                    res.append(base + _VOWEL_SIGNS[chars[i + 1]])
                    i += 1
                else:
                    if i + 1 == n or chars[i + 1] in " \t\n,.-/":
                        res.append(base)
                    else:
                        res.append(base + "a")
            else:
                res.append(c)
            i += 1
        word_out = "".join(res)
        if word_out:
            output_words.append(word_out.capitalize())
    return " ".join(output_words)

def translate_indic_phrase_to_english(text: str) -> str:
    """Semantic translation for administrative terms, boundaries, and land categories."""
    if not text:
        return ""
    clean = text.strip()
    if clean in SEMANTIC_CATEGORIES_MAP:
        return SEMANTIC_CATEGORIES_MAP[clean]

    # Handle boundary phrases like "उत्तर: ग्रामीण रास्ता, दक्षिण: खसरा 889/5"
    t = clean
    replacements = {
        "उत्तर": "North", "दक्षिण": "South", "पूर्व": "East", "पश्चिम": "West",
        "ग्रामीण रास्ता": "Village Road", "रास्ता": "Road / Path", "सड़क": "Road",
        "मार्ग": "Road / Marg", "सेक्टर मार्ग": "Sector Road", "नहर": "Canal",
        "नहर शाखा": "Canal Branch", "कुआं": "Well", "खेत": "Field / Farm",
        "खुला खेत": "Open Field", "बगीचा": "Garden / Orchard", "प्लॉट": "Plot",
        "खसरा": "Khasra", "खाता": "Khata", "बीघा": "Bigha"
    }
    for hi, en in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
        t = t.replace(hi, en)
    return t

def has_indic_script(text: str) -> bool:
    if not text:
        return False
    return any(0x0900 <= ord(c) <= 0x097F for c in text)

def normalize_multilingual_text(raw_text: str) -> Tuple[str, str]:
    """Reconstructs ligatures, normalizes numerals, swaps visual matras, and detects document language ('hi' or 'en')."""
    if not raw_text:
        return "", "en"

    import unicodedata
    t = raw_text.translate(_HOMOGLYPH_MAP).translate(_DEVANAGARI_DIGITS)
    for lig in sorted(_LIGATURE_REPLACEMENTS.keys(), key=len, reverse=True):
        t = t.replace(lig, _LIGATURE_REPLACEMENTS[lig])

    # Correct visual-order chhoti-i matra (\u093f) before consonant to logical order after consonant
    t = re.sub(r"(?<![\u0915-\u0939\u094d])\u093f([\u0915-\u0939](?:\u094d[\u0915-\u0939])*)", r"\g<1>" + "\u093f", t)
    t = unicodedata.normalize("NFC", t)

    lang = "hi" if has_indic_script(t) else "en"
    return t, lang

_IDENTIFIER_LABEL_WORDS = {
    "no", "no.", "number", "n0", "n0.", "na", "n/a", "-", "nil", "none", "null",
    "num", "id", "code", "details", "particulars", "new", "old", "and", "or",
    "commercial", "agricultural", "residential", "khata", "khasra", "plot", "sector",
    "property", "survey", "gata", "identifiers", "location", "नं", "नं.", "संख्या",
    "पहचानकर्ता", "विवरण"
}

# ----------------- OCR PREPROCESSING & RUNNERS -----------------

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
    global _paddle_engine
    if _paddle_engine is None:
        with _paddle_lock:
            if _paddle_engine is None:
                # Try devanagari / hi first for multilingual support, fall back to en if model not cached
                for target_lang in (PADDLE_LANG, "hi", "en"):
                    try:
                        _paddle_engine = PaddleOCR(
                            lang=target_lang,
                            use_doc_orientation_classify=PADDLE_DOC_ORIENTATION,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                            enable_mkldnn=False,
                        )
                        break
                    except Exception:
                        continue
                if _paddle_engine is None:
                    _paddle_engine = PaddleOCR(lang="en", use_angle_cls=False, enable_mkldnn=False)
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

# ----------------- BILINGUAL CANONICAL NORMALIZER & PROMPT -----------------

_CANONICAL_PROMPT = """You are a multilingual Land Record Canonical Normalizer for Indian Land Records (Department of Land Resources).
Extract and structure the document into this exact key-value template.
Support both English and Hindi/Indic documents. If the document is in Hindi or regional language:
- Keep the original native script for Native fields and provide the accurate English transliteration/translation for the standard fields.
- Convert dates to DD-MM-YYYY format.
- Total Area should be a numeric value (Bigha or Sq. Yards).
- Consideration should be numeric digits only.

Exact Template:
DOCUMENT TYPE: <Sale Deed | Succession / Mutation | Khasra-Khatauni Record | Record of Rights (RoR) | Patta / Lease>
DOCUMENT TYPE NATIVE: <विक्रय विलेख | नामांतरण | जमाबंदी / खतौनी | पट्टा>
SERIAL NO: <deed/serial/stamp number>
District: <district in English>
District Native: <district in native script>
Tehsil: <tehsil in English>
Tehsil Native: <tehsil in native script>
Village: <village/locality in English>
Village Native: <village/locality in native script>
Khasra No: <khasra number>
Khata No: <khata number>
Plot No: <plot number>
Property ID: <property ID>
Vendor: <seller name in English>
Vendor Native: <seller name in native script>
Recorded Holder: <buyer/recorded owner in English>
Recorded Holder Native: <buyer/recorded owner in native script>
Share / Hissa: <share e.g. 1/1 (Full)>
Total Area: <numeric area>
Land Use: <Agricultural | Commercial (Converted) | Residential>
Land Use Native: <कृषि | व्यावसायिक | आवासीय>
Consideration: <numeric amount>
Registration Date: <DD-MM-YYYY>
Mutation Date: <DD-MM-YYYY>
Boundaries: North: <north> | South: <south> | East: <east> | West: <west>
Boundaries Native: <उत्तर: ... | दक्षिण: ... | पूर्व: ... | पश्चिम: ...>
"""

def _call_gemini_normalizer(raw_text: str, image_png_bytes: Optional[bytes] = None) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        parts: List[Dict[str, Any]] = [{"text": _CANONICAL_PROMPT}]
        if raw_text and len(raw_text.strip()) > 20:
            parts.append({"text": f"\nRAW TEXT:\n{raw_text[:8000]}"})
        if image_png_bytes:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_png_bytes).decode("ascii")
                }
            })
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 700}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            cand_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if "Khasra No:" in cand_text or "District:" in cand_text:
                return cand_text
    except Exception as e:
        print(f"[Normalizer] Gemini fallback ({e})")
    return None

def _local_normalize_text(raw_text: str) -> str:
    """Pre-processes multi-column tables, breaks inline sentences onto separate lines."""
    if not raw_text:
        return ""
    t, _ = normalize_multilingual_text(raw_text)

    # Break inline narrative sentences onto separate lines
    t = re.sub(
        r"\.\s+(?=(?:Vendee|Vendor|Buyer|Seller|Recorded\s+Holder|Claimant|Location|Identifiers|Khasra|Khata|Plot|Total\s+Area|Consideration|Registration|Mutation|Boundaries|विक्रेता|क्रेता|अभिलेखित|खसरा|खाता|प्लॉट|क्षेत्रफल|प्रतिफल|पंजीयन|नामांतरण|चौहद्दी)\b)",
        "\n",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\s+(\d+\.\s*(?:Location|Identifier|Person|Property|Land|Owner|History|Registration|Boundary|स्थान|पहचानकर्ता|व्यक्ति|विवरण|इतिहास)(?:\s*(?:Details|Particulars))?)",
        r"\n\1\n",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\s+(?=(?:Contact|Claimed\s*Share|हिस्सा|भाग)\s*:)", "\n", t, flags=re.IGNORECASE)

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    unpivoted = []
    for idx in range(len(lines) - 1):
        h_parts = [p.strip() for p in lines[idx].split("|") if p.strip()]
        v_parts = [p.strip() for p in lines[idx + 1].split("|") if p.strip()]
        if len(h_parts) >= 2 and len(h_parts) == len(v_parts):
            for hp, vp in zip(h_parts, v_parts):
                if ":" not in hp:
                    unpivoted.append(f"{hp}: {vp}")

    if unpivoted:
        t = t + "\n---UNPIVOTED_TABLE---\n" + "\n".join(unpivoted)
    return t

def run_ocr(file_content: bytes, filename: str) -> Tuple[str, float]:
    file_hash = hashlib.sha256(file_content).hexdigest()
    if file_hash in _OCR_CACHE and _OCR_CACHE[file_hash][0]:
        return _OCR_CACHE[file_hash]

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    text = ""
    quality = 0.90
    first_page_png: Optional[bytes] = None

    if ext == "txt":
        raw_t = file_content.decode("utf-8", errors="ignore").strip()
        norm_t = _local_normalize_text(raw_t)
        res = (norm_t + "\n" + raw_t, 1.0)
        _OCR_CACHE[file_hash] = res
        return res

    if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"):
        first_page_png = file_content
        gemini_text = _call_gemini_normalizer("", first_page_png)
        if gemini_text:
            res = (gemini_text, 0.97)
            _OCR_CACHE[file_hash] = res
            return res

        if OCR_AVAILABLE:
            try:
                text, conf = _ocr_image_bytes(file_content)
                quality = round(min(0.99, conf), 2) if conf is not None else 0.88
            except Exception as e:
                print(f"[OCR] Image OCR error: {e}")
                quality = 0.0

    elif ext == "pdf":
        page_pngs = []
        if FITZ_AVAILABLE:
            try:
                doc = fitz.open(stream=file_content, filetype="pdf")
                text = _extract_pdf_structured_text(doc)
                if len(doc) > 0:
                    first_page_png = doc[0].get_pixmap(dpi=150).tobytes("png")
                if len(text.strip()) < 60 and OCR_AVAILABLE:
                    dpi = 150 if OCR_ENGINE == "paddle" else 220
                    for idx in range(min(len(doc), MAX_OCR_PAGES)):
                        page_pngs.append(doc[idx].get_pixmap(dpi=dpi).tobytes("png"))
                doc.close()
            except Exception as e:
                print(f"[PDF] PyMuPDF extraction error: {e}")

        if len(text.strip()) < 60 and PDF_TEXT_AVAILABLE and not page_pngs:
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_OCR_PAGES])
            except Exception:
                text = ""

        if page_pngs and OCR_AVAILABLE and not (GEMINI_API_KEY and first_page_png):
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
                    pass
            ocr_text = "\n".join(ocr_parts).strip()
            if len(ocr_text) > len(text.strip()):
                text = ocr_text
                quality = round(min(0.96, sum(confs) / len(confs)), 2) if confs else 0.82
        elif text.strip():
            quality = 0.96

    ai_canonical = _call_gemini_normalizer(text, first_page_png)
    local_canonical = _local_normalize_text(text)
    final_text = (ai_canonical + "\n---RAW_DOCUMENT---\n" + local_canonical) if ai_canonical else local_canonical

    res = (final_text.strip(), quality)
    if res[0]:
        _OCR_CACHE[file_hash] = res
    return res

# ----------------- BILINGUAL FIELD CLEANERS -----------------

def clean_extracted_name(raw: str) -> Tuple[str, str]:
    """Cleans names in both English and Indic scripts, removing honorifics and genealogical suffixes."""
    if not raw:
        return "", ""
    s, lang = normalize_multilingual_text(raw)

    # 1. Strip role titles (English & Hindi)
    s = re.sub(
        r"(?i)^(?:(?:\d+(?:st|nd|rd|th)?\s*)?(?:Recorded\s*(?:Holder|Owner)|Legal\s*Heir|Owner|Vendor|Vendee|Seller|Buyer|Prior\s*Transferor|Transferor|Claimant|विक्रेता|क्रेता|खरीदार|अभिलेखित\s*धारक|खातेदार|धारक)\s*(?:\([^\)]*\))?\s*[:\-]?\s*)+",
        "",
        s,
    ).strip()

    # 2. Strip honorifics (English & Hindi)
    s = re.sub(r"(?i)^(?:late\s+)?(?:shri|smt\.?|mr\.?|mrs\.?|ms\.?|dr\.?|kumari)\s+", "", s)
    s = re.sub(r"^(?:श्रीमान|श्रीमती|श्री|स्व\.?|स्वर्गीय|सुश्री|डॉ\.?)\s+", "", s)

    # 3. Strip genealogical suffixes and relationships
    s = re.sub(
        r"(?i)\b(?:s/o|d/o|w/o|c/o|aged|age\b|r/o|resident\s+of|residing\s+at|son\s+of|daughter\s+of|wife\s+of|widow\s+of|\(deceased\)|deceased|legal\s+heir|caste|occupation|contact|mobile|claimed\s+share|share|hissa|hereinafter)\b.*",
        "",
        s,
    )
    s = re.sub(
        r"(?:आत्मज|सुपुत्र|पुत्र|सुपुत्री|पुत्री|पत्नी|बेवा|निवासी|उम्र|आयु|जाति|मोबाइल|संपर्क|हिस्सा|भाग)\b.*",
        "",
        s,
    )

    s = re.sub(r"[\.,;:_~\(\)\[\]\|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")

    if not s or s.lower() in _IDENTIFIER_LABEL_WORDS:
        return "", ""

    if lang == "hi":
        native_val = s
        en_val = transliterate_indic_to_english(s)
    else:
        words = []
        for w in s.split():
            if w.upper() in ("LLP", "PVT", "LTD", "INC", "JDA", "RIICO", "HUF", "CO") or "&" in w:
                words.append(w.upper())
            else:
                words.append(w.capitalize())
        en_val = " ".join(words)
        native_val = ADMIN_LOCATIONS_REVERSE.get(en_val.lower(), en_val)

    return en_val, native_val

def clean_location_value(raw: Optional[str], is_village: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Cleans location names and returns (English, Native) pair."""
    if not raw:
        return None, None
    s, lang = normalize_multilingual_text(raw)

    for _ in range(3):
        s = re.sub(
            r"^(?:[\d]+(?:st|nd|rd|th)?\s*)?(?:\/\s*)?(?:Urban\s+Locality|Revenue\s+Village|Locality|Village|Sector|Location|District|Tehsil|Gram|Mauza|स्थान|जिला|तहसील|ग्राम|गाँव|मौजा|क्षेत्र)\s*(?:\/\s*(?:Urban\s+)?(?:Locality|Sector|Village))?\s*[:\-\.]*\s*",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

    s = re.split(
        r"(?i)(?:\b\d+\.\s*(?:Location|Identifier|Person|Property|Land|Owner|History|Registration|Boundary|Deed|स्थान|पहचानकर्ता|व्यक्ति|विवरण|इतिहास)|\b(?:IDENTIFIERS?|Location|Vendee|Vendor|Buyer|Seller|Recorded|Claimant|Tehsil|District|Khasra|Khata|Plot|Total\s*Area|विक्रेता|क्रेता|अभिलेखित|खसरा|खाता|प्लॉट|क्षेत्रफल|पहचानकर्ता)\b)",
        s,
    )[0]

    s = s.split(".")[0]
    s = re.sub(r"\s+\d+$", "", s)
    s = re.sub(r"[\|\n\r]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-/()")

    if not s or len(s) < 2 or s.lower() in _IDENTIFIER_LABEL_WORDS:
        return None, None

    if lang == "hi":
        native_val = s
        en_val = ADMIN_LOCATIONS_MAP.get(s, transliterate_indic_to_english(s))
    else:
        en_val = s.title()
        native_val = ADMIN_LOCATIONS_REVERSE.get(en_val.lower(), en_val)

    return en_val, native_val

def _clean_identifier_match(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    val, _ = normalize_multilingual_text(raw)
    val = re.sub(r"^(?:No\.?|Number|Num|ID|#|नं\.?|संख्या)\s*[:\-\.]?\s*", "", val, flags=re.IGNORECASE).strip()
    val = val.strip(":-.,;|()[]{}")
    if not val or val.lower().rstrip(".") in _IDENTIFIER_LABEL_WORDS:
        return None
    if not re.search(r"[0-9]", val):
        return None
    return val

def normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s, _ = normalize_multilingual_text(raw)
    s = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    # Hindi month names mapping
    hindi_months = {
        "जनवरी": "January", "फरवरी": "February", "मार्च": "March", "अप्रैल": "April",
        "मई": "May", "जून": "June", "जुलाई": "July", "अगस्त": "August",
        "सितंबर": "September", "अक्टूबर": "October", "नवंबर": "November", "दिसंबर": "December"
    }
    for hi, en in hindi_months.items():
        s = re.sub(rf"\b{hi}\b", en, s)

    s = re.sub(r"[,\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for fmt in (
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d %B %Y", "%d %b %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y",
        "%B %d %Y", "%b %d %Y", "%d-%m-%y", "%d/%m/%y"
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
    s, _ = normalize_multilingual_text(raw)
    s = s.strip()
    if re.search(r"\b0\s*\/\-|\b0\.00\b|^0$", s):
        return 0.0
    mult = 1.0
    if re.search(r"\b(?:crore|crores|cr\.?|करोड़)\b", s, re.IGNORECASE):
        mult = 10000000.0
    elif re.search(r"\b(?:lakh|lakhs|lac|lacs|लाख)\b", s, re.IGNORECASE):
        mult = 100000.0
    num_m = re.search(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", s)
    if not num_m:
        return None
    cleaned = num_m.group(1).replace(",", "")
    try:
        return round(float(cleaned) * mult, 2)
    except ValueError:
        return None

# ----------------- MULTILINGUAL LAND IDENTITY EXTRACTOR -----------------

def extract_land_identity(text: str):
    result = {
        "source_language": "en",
        "document_type": "Sale Deed",
        "document_type_native": "विक्रय विलेख (Sale Deed)",
        "serial_number": None,
        "district": None,
        "district_native": None,
        "tehsil": None,
        "tehsil_native": None,
        "village": None,
        "village_native": None,
        "khasra_number": None,
        "khata_number": None,
        "plot_number": None,
        "property_id": None,
        "survey_number": None,
        "owner_name": None,
        "owner_name_native": None,
        "seller_name": None,
        "seller_name_native": None,
        "buyer_name": None,
        "buyer_name_native": None,
        "share_hissa": "1/1 (Full)",
        "total_area": None,
        "sub_plot_areas": [],
        "land_use": "Agricultural",
        "land_use_native": "कृषि",
        "boundaries_desc": None,
        "boundaries_desc_native": None,
        "consideration_amount": None,
        "registration_date": None,
        "mutation_date": None,
    }
    confidence: Dict[str, float] = {}
    if not text:
        return result, confidence

    t = _local_normalize_text(text)
    is_hindi = has_indic_script(t)
    result["source_language"] = "hi" if is_hindi else "en"
    upper = t.upper()

    # Document Type (Bilingual)
    if "SUCCESSION" in upper or "FAUTI" in upper or "नामांतरण" in t or "फौती" in t:
        result["document_type"] = "Succession / Mutation"
        result["document_type_native"] = "नामांतरण (Succession / Mutation)"
    elif "KHATAUNI" in upper or "JAMABANDI" in upper or "जमाबंदी" in t or "खतौनी" in t:
        result["document_type"] = "Khasra-Khatauni Record"
        result["document_type_native"] = "जमाबंदी / खतौनी (RoR)"
    elif "RECORD OF RIGHTS" in upper or re.search(r"\bROR\b", upper) or "अधिकार अभिलेख" in t:
        result["document_type"] = "Record of Rights (RoR)"
        result["document_type_native"] = "अधिकार अभिलेख (RoR)"
    elif "PATTA" in upper or "LEASE" in upper or "पट्टा" in t or "लीज" in t:
        result["document_type"] = "Patta / Lease"
        result["document_type_native"] = "पट्टा / लीज (Patta / Lease)"
    elif "हस्तांतरण" in t:
        result["document_type"] = "Transfer Deed"
        result["document_type_native"] = "हस्तांतरण विलेख"
    elif "CONVEYANCE" in upper or "SALE DEED" in upper or "बैनामा" in t or "विक्रय विलेख" in t or "विक्रय अभिलेख" in t:
        result["document_type"] = "Sale Deed"
        result["document_type_native"] = "विक्रय विलेख (Sale Deed)"

    # 1. Serial / Deed Number (Bilingual)
    serial_patterns = [
        r"(?:Deed|Serial|Stamp|e-Stamp|Document|Certificate|क्रमांक\s*संख्या|पंजीयन\s*संख्या|दस्तावेज\s*क्रमांक)\s*(?:No\.?|N0\.?|Number|#|सं\.?)?\s*[:\.\-]\s*([A-Z0-9][A-Z0-9\-\/]{3,24})",
        r"\bDeed\s*#\s*([A-Z0-9\-\/]{4,24})",
        r"\b((?:MUT|RJ|IN-RJ|SD|DEED|UDH)[\-][A-Z0-9\-]{4,24})\b",
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

    # 2. Registration Date & Mutation Date (Bilingual)
    date_tok = r"([0-9]{1,2}[\-\/\s][A-Za-z0-9\u0900-\u097F]{2,12}[\-\/\s][0-9]{2,4})"
    reg_patterns = [
        r"(?:Registration\s*Date|पंजीयन\s*दिनांक|रजिस्ट्री\s*तारीख)\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
        r"(?:SRO[^\n\r]*?|Deed[^\n\r]*?)\bDated\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
        r"(?<![\u0900-\u097F])दिनांक(?![\u0900-\u097F])\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
        r"\bDated\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
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
        r"(?:Mutation\s*Date|नामांतरण\s*दिनांक|आदेश\s*दिनांक|दाखिल\s*खारिज\s*तारीख)\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
        r"\bOrder\s*Date\s*[:\-]?\s*(?:\n\s*)?" + date_tok,
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

    # 3. Location: District, Tehsil, Village (Bilingual)
    m_dist = re.search(r"(?<![\u0900-\u097F])(?:District|Distt\.?|Zila|जिला|ज़िला)(?![\u0900-\u097F])\s*[:\-]\s*([A-Za-z\u0900-\u097F\s]{2,25}?)(?:,|\n|\||\.|Tehsil|Village|तहसील|ग्राम|$)", t, re.IGNORECASE)
    if m_dist:
        d_en, d_nat = clean_location_value(m_dist.group(1), is_village=False)
        if d_en:
            result["district"] = d_en
            result["district_native"] = d_nat
            confidence["district"] = 0.98

    if not result["district"]:
        for hi_d, en_d in ADMIN_LOCATIONS_MAP.items():
            if re.search(rf"\b{re.escape(hi_d)}\b|\b{re.escape(en_d)}\b", t, re.IGNORECASE):
                result["district"] = en_d
                result["district_native"] = hi_d
                confidence["district"] = 0.94
                break

    m_teh = re.search(r"(?<![\u0900-\u097F])(?:Tehsil|Tahsil|Taluka|तहसील|तालुका)(?![\u0900-\u097F])\s*[:\-]\s*([A-Za-z\u0900-\u097F\s]{2,25}?)(?:,|\n|\||\.|Village|Urban|Locality|Sector|District|ग्राम|गाँव|$)", t, re.IGNORECASE)
    if m_teh:
        t_en, t_nat = clean_location_value(m_teh.group(1), is_village=False)
        if t_en:
            result["tehsil"] = t_en
            result["tehsil_native"] = t_nat
            confidence["tehsil"] = 0.96
    elif result["district"]:
        result["tehsil"] = result["district"]
        result["tehsil_native"] = result["district_native"]
        confidence["tehsil"] = 0.85

    if not result["district"] and result["tehsil"]:
        if result["tehsil_native"] in ADMIN_LOCATIONS_MAP or result["tehsil"] in ADMIN_LOCATIONS_MAP.values():
            result["district"] = result["tehsil"]
            result["district_native"] = result["tehsil_native"]
            confidence["district"] = 0.90

    vill_patterns = [
        r"\b(?:Village\s*\/\s*(?:Urban\s*)?(?:Locality|Sector)|Urban\s*Locality|Revenue\s*Village|Village|Locality|Gram|Mauza|ग्राम|गाँव|मौजा)\s*[:\-]\s*([^\n\r\|]+(?:\n(?![A-Z\u0900-\u097F\s]{4,}(?:NUMBER|NO|RECORD|IDENTIFIER|PERSON|LOCATION|पहचानकर्ता|व्यक्ति))[^\n\r\|]+)?)",
        r"(?:situated|located|lying|स्थित)\s+(?:at|in|में)?\s*(?:Village|Gram|Mauza|Locality|ग्राम|गाँव)?\s*([A-Za-z0-9\u0900-\u097F\s\-]{2,35}?)(?:,|\.|Tehsil|District|तहसील|जिला|\n)",
    ]
    for vp in vill_patterns:
        for m_v in re.finditer(vp, t, re.IGNORECASE):
            v_en, v_nat = clean_location_value(m_v.group(1), is_village=True)
            if v_en:
                result["village"] = v_en
                result["village_native"] = v_nat
                confidence["village"] = 0.95
                break
        if result["village"]:
            break

    # 4. Identifiers: Khata, Khasra, Plot, Property ID (Bilingual)
    khata_patterns = [
        r"(?:KHATA\s*NUMBER|खाता\s*(?:नं\.?|संख्या))\s*[:\-]?\s*(?:\n\s*)?(?:Commercial\s+Khata\s+|Agricultural\s+Khata\s+|Khata\s+)?([A-Z0-9][A-Z0-9\/\-]*)",
        r"\bKhata\s*(?:No\.?|Number)?\s*[:\-]\s*(?:Commercial\s+Khata\s+|Khata\s+)?([A-Z0-9][A-Z0-9\/\-]*)",
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
        r"(?:KHASRA\s*NUMBER|खसरा\s*(?:नं\.?|संख्या)|सर्वे\s*(?:नं\.?|संख्या))\s*[:\-]?\s*(?:\n\s*)?(?:Khasra\s+)?([0-9][0-9\/\-A-Za-z]*)",
        r"\b(?:Khasra|Survey|Gata|Killa)\s*(?:No\.?|Number)?\s*[:\-]\s*(?:Khasra\s+)?([0-9][0-9\/\-A-Za-z]*)",
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
        r"(?:Plot\s*(?:No\.?|Number)|प्लॉट\s*(?:नं\.?|संख्या)|भूखंड\s*(?:नं\.?|संख्या))\s*[:\-\.]?\s*([A-Za-z0-9][A-Za-z0-9\-\/]*(?:,\s*Sector\s*[A-Za-z0-9]+)?)",
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
        r"\b((?:UPID|ULPIN|PID|PROP|UDH)[\-][A-Z0-9\-_\/]{3,28})\b",
        r"(?:Property\s*ID|संपत्ति\s*आईडी)\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9\-_\/]{2,24})",
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

    # 5. Holding: Buyer/Owner, Seller, Share (Bilingual)
    owner_patterns = [
        r"(?:RECORDED\s*HOLDER|अभिलेखित\s*धारक|(?<!वि)क्रेता|खरीदार|खातेदार|BUYER|VENDEE)\s*(?:\([^\)]*\))?\s*[:\-]?\s*(?:\n\s*)?([A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F\s\.\&\-]{2,60}?)(?:,|\n|\||SELLER|VENDOR|विक्रेता|S\/O|D\/O|W\/O|आत्मज|पुत्र|पत्नी|हिस्सा|$)",
    ]
    for op in owner_patterns:
        for m in re.finditer(op, t, re.IGNORECASE):
            en_name, nat_name = clean_extracted_name(m.group(1))
            if len(en_name) >= 3:
                result["buyer_name"] = en_name
                result["buyer_name_native"] = nat_name
                confidence["buyer_name"] = 0.94
                break
        if result["buyer_name"]:
            break

    seller_patterns = [
        r"(?:SELLER|VENDOR|विक्रेता|बेचने\s*वाला)\s*(?:\([^\)]*\))?\s*[:\-]?\s*(?:\n\s*)?([A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F\s\.\&\-]{2,60}?)(?:,|\n|\||SHARE|HISSA|हिस्सा|RECORDED|(?<!वि)क्रेता|अभिलेखित|S\/O|D\/O|W\/O|आत्मज|पुत्र|$)",
    ]
    for sp in seller_patterns:
        for m in re.finditer(sp, t, re.IGNORECASE):
            en_name, nat_name = clean_extracted_name(m.group(1))
            if len(en_name) >= 3:
                result["seller_name"] = en_name
                result["seller_name_native"] = nat_name
                confidence["seller_name"] = 0.93
                break
        if result["seller_name"]:
            break

    result["owner_name"] = result["buyer_name"] or result["seller_name"]
    result["owner_name_native"] = result["buyer_name_native"] or result["seller_name_native"]
    if result["owner_name"]:
        confidence["owner_name"] = confidence.get("buyer_name", confidence.get("seller_name", 0.92))

    m_sh = re.search(r"(?:SHARE\s*(?:\/\s*HISSA)?|CLAIMED\s*SHARE|(?<![\u0900-\u097F])हिस्सा(?![\u0900-\u097F])|(?<![\u0900-\u097F])भाग(?![\u0900-\u097F]))\s*[:\-]\s*([0-9]+\/[0-9]+(?:\s*(?:भाग|share|\([^\)]*\)))?|पूर्ण|full(?:\s*inheritance)?|[^\n\r,\|]+)", t, re.IGNORECASE)
    if m_sh:
        sh_raw = re.sub(r"\s+", " ", m_sh.group(1)).strip()
        result["share_hissa"] = translate_indic_phrase_to_english(sh_raw)
        confidence["share_hissa"] = 0.96

    # 6. Total Area (Bilingual)
    m_area = re.search(r"(?:Total\s*Area|Area|कुल\s*क्षेत्रफल|क्षेत्रफल|रकबा)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:bigha|बीघा|वर्ग\s*गज|हेक्टेयर|sq\.?\s*yards?)?", t, re.IGNORECASE)
    if m_area and float(m_area.group(1)) > 0:
        result["total_area"] = float(m_area.group(1))
        confidence["total_area"] = 0.96

    sub_matches = re.findall(r"(?:Sub-plot|उप-प्लॉट)\s*[A-Za-z0-9]*\s*=\s*([0-9\.]+)", t, re.IGNORECASE)
    if sub_matches:
        result["sub_plot_areas"] = [float(x) for x in sub_matches]

    # 7. Land Use (Bilingual)
    m_lu = re.search(r"(?:Land\s*Use|भूमि\s*उपयोग|किस्म\s*भूमि)\s*[:\-]\s*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s\(\)\/]+?)(?:,|\n|\||\.|Consideration|Total|प्रतिफल|$)", t, re.IGNORECASE)
    if m_lu:
        lu_raw = m_lu.group(1).strip()
        result["land_use_native"] = lu_raw
        result["land_use"] = translate_indic_phrase_to_english(lu_raw)
        confidence["land_use"] = 0.94
    elif "आवासीय" in t or "RESIDENTIAL" in upper:
        result["land_use"] = "Residential"
        result["land_use_native"] = "आवासीय"
    elif "कृषि" in t or "AGRICULTURAL" in upper:
        result["land_use"] = "Agricultural"
        result["land_use_native"] = "कृषि"

    # 8. Consideration Value (Bilingual)
    cons_patterns = [
        r"(?:CONSIDERATION\s*(?:VALUE|AMOUNT)?|प्रतिफल\s*राशि|विक्रय\s*मूल्य)\s*[:\-]?\s*(?:Rs\.?|INR|₹|रु[\.\s]*)*[\n\r\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?|0\/\-|0)",
        r"(?:Sale\s*Price|Sale\s*Value|Consideration|कीमत)\s*[:\-]\s*(?:Rs\.?|INR|₹|रु[\.\s]*)*[\n\r\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
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

    # 9. Boundaries / Chauhaddi (Bilingual)
    dir_parts = []
    dir_parts_native = []
    bilingual_dirs = [
        ("North", r"(?:North|उत्तर)"),
        ("South", r"(?:South|दक्षिण|दिक्षण|दकि्षण)"),
        ("East", r"(?:East|पूर्व)"),
        ("West", r"(?:West|पश्चिम|पिश्चिम|पशि्चम)")
    ]
    for dir_en, pat in bilingual_dirs:
        dm = re.search(
            pat + r"\s*(?:\([^\)]*\))?\s*[:\-–]\s*([^\|\n\r,]+(?:\n(?!\s*(?:North|South|East|West|उत्तर|दक्षिण|पूर्व|पश्चिम|\d+\.|\-\-\-))[^\|\n\r,]+)?)",
            t,
            re.IGNORECASE,
        )
        if dm:
            val_d = re.sub(r"\s+", " ", dm.group(1)).strip(" ,;.|")
            val_d = re.sub(r"\s*(?:\d+\.\s*(?:इतिहास|History|पंजीयन|Registration)|\-\-\-(?:CELL_PAIRS|UNPIVOTED_TABLE|RAW_DOCUMENT)\-\-\-).*$", "", val_d).strip()
            if val_d:
                dir_parts_native.append(f"{dir_en}: {val_d}")
                dir_parts.append(f"{dir_en}: {translate_indic_phrase_to_english(val_d)}")

    if len(dir_parts) >= 2:
        result["boundaries_desc"] = " | ".join(dir_parts)
        result["boundaries_desc_native"] = " | ".join(dir_parts_native)
        confidence["boundaries_desc"] = 0.94
    else:
        m_b = re.search(r"(?:BOUNDARIES|चौहद्दी|सीमाएं)[^\n]*[:\-]\s*([^\n\r]+)", t, re.IGNORECASE)
        if m_b:
            b_raw = m_b.group(1).strip()
            result["boundaries_desc_native"] = b_raw
            result["boundaries_desc"] = translate_indic_phrase_to_english(b_raw)
            confidence["boundaries_desc"] = 0.90

    return result, confidence

# ----------------- VALIDATION RULES -----------------

def run_validation_rules(data: dict, conn: sqlite3.Connection, current_record_id: Optional[int] = None) -> List[str]:
    flags = []

    if not data.get("khasra_number") and not data.get("plot_number"):
        flags.append("Incomplete Extraction: Missing Khasra / Plot Identifier")
    if not data.get("owner_name") and not data.get("owner_name_native"):
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

    if not data.get("district") and not data.get("district_native"):
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
              AND (LOWER(TRIM(village)) = LOWER(TRIM(?)) OR LOWER(TRIM(COALESCE(village_native, ''))) = LOWER(TRIM(?)))
              AND (LOWER(TRIM(district)) = LOWER(TRIM(?)) OR LOWER(TRIM(COALESCE(district_native, ''))) = LOWER(TRIM(?)))
        """
        params = [khasra, village, village, district, district]
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
    else:
        final_status = "Verified" if len(flags) == 0 else "Flagged"

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO records (
            filename, file_path, status, raw_text, source_language,
            owner_name, owner_name_native, seller_name, seller_name_native,
            buyer_name, buyer_name_native, share_hissa, document_type, document_type_native,
            serial_number, survey_number, khasra_number, khata_number, plot_number,
            property_id, total_area, sub_plot_areas, land_use, land_use_native,
            boundaries_desc, boundaries_desc_native, village, village_native,
            tehsil, tehsil_native, district, district_native, consideration_amount,
            registration_date, mutation_date, confidence_scores, validation_flags,
            uploaded_by, ocr_quality, submission_source, submitted_by_user_id,
            verification_deadline, has_conflict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        file_location,
        final_status,
        raw_text,
        extracted["source_language"],
        extracted["owner_name"],
        extracted["owner_name_native"],
        extracted["seller_name"],
        extracted["seller_name_native"],
        extracted["buyer_name"],
        extracted["buyer_name_native"],
        extracted["share_hissa"],
        extracted["document_type"],
        extracted["document_type_native"],
        extracted["serial_number"],
        extracted["survey_number"],
        extracted["khasra_number"],
        extracted["khata_number"],
        extracted["plot_number"],
        extracted["property_id"],
        extracted["total_area"],
        json.dumps(extracted["sub_plot_areas"]),
        extracted["land_use"],
        extracted["land_use_native"],
        extracted["boundaries_desc"],
        extracted["boundaries_desc_native"],
        extracted["village"],
        extracted["village_native"],
        extracted["tehsil"],
        extracted["tehsil_native"],
        extracted["district"],
        extracted["district_native"],
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

    return {
        "id": record_id,
        "filename": filename,
        "status": final_status,
        "source_language": extracted["source_language"],
        "owner_name": extracted["owner_name"],
        "owner_name_native": extracted["owner_name_native"],
        "khasra_number": extracted["khasra_number"],
        "village": extracted["village"],
        "flags": flags,
        "has_conflict": bool(has_conflict)
    }

# ----------------- API ENDPOINTS -----------------

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
                   f"Ingested record #{result['id']} ({file.filename}) - Lang: {result['source_language']}, Result: {result['status']}")
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
                   f"Record #{row['id']} missed its physical-verification deadline and was escalated.")
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
            rec["owner_name_native"] = "[RESTRICTED]"
            rec["seller_name"] = "[RESTRICTED]"
            rec["seller_name_native"] = "[RESTRICTED]"
            rec["buyer_name"] = "[RESTRICTED]"
            rec["buyer_name_native"] = "[RESTRICTED]"
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
        SELECT id, khasra_number, village, village_native, district, district_native,
               seller_name, seller_name_native, buyer_name, buyer_name_native,
               registration_date, status
        FROM records
        WHERE khasra_number IS NOT NULL AND TRIM(khasra_number) != ''
          AND (village IS NOT NULL OR village_native IS NOT NULL)
        ORDER BY id
    """).fetchall()

    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        rec = dict(r)
        key = (
            (rec["khasra_number"] or "").strip().lower(),
            (rec["village"] or rec.get("village_native") or "").strip().lower(),
            (rec["district"] or rec.get("district_native") or "").strip().lower(),
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
                prev_buyer = (recs_sorted[i - 1].get("buyer_name") or recs_sorted[i - 1].get("buyer_name_native") or "").strip().lower()
                cur_seller = (rec.get("seller_name") or rec.get("seller_name_native") or "").strip().lower()
                if prev_buyer and cur_seller and prev_buyer != cur_seller:
                    continuity_ok = False
                    has_break = True
            hops.append({
                "record_id": rec["id"],
                "seller": rec.get("seller_name") or rec.get("seller_name_native") or "Unknown",
                "seller_native": rec.get("seller_name_native"),
                "buyer": rec.get("buyer_name") or rec.get("buyer_name_native") or "Unknown",
                "buyer_native": rec.get("buyer_name_native"),
                "date": rec.get("registration_date"),
                "status": rec.get("status"),
                "continuity_ok": continuity_ok,
            })

        chains.append({
            "khasra_number": recs_sorted[0]["khasra_number"],
            "village": recs_sorted[0]["village"] or recs_sorted[0].get("village_native"),
            "village_native": recs_sorted[0].get("village_native"),
            "district": recs_sorted[0]["district"] or recs_sorted[0].get("district_native"),
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
    district_native: Optional[str] = None
    tehsil: Optional[str] = None
    tehsil_native: Optional[str] = None
    village: Optional[str] = None
    village_native: Optional[str] = None
    khasra_number: Optional[str] = None
    khata_number: Optional[str] = None
    plot_number: Optional[str] = None
    property_id: Optional[str] = None
    owner_name: str
    owner_name_native: Optional[str] = None
    seller_name: Optional[str] = None
    seller_name_native: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_name_native: Optional[str] = None
    share_hissa: Optional[str] = None
    total_area: Optional[float] = None
    land_use: Optional[str] = None
    land_use_native: Optional[str] = None
    boundaries_desc: Optional[str] = None
    boundaries_desc_native: Optional[str] = None
    consideration_amount: Optional[float] = None
    registration_date: Optional[str] = None
    mutation_date: Optional[str] = None
    serial_number: Optional[str] = None
    document_type: Optional[str] = None
    document_type_native: Optional[str] = None
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
            district = ?, district_native = ?, tehsil = ?, tehsil_native = ?,
            village = ?, village_native = ?, khasra_number = ?, khata_number = ?,
            plot_number = ?, property_id = ?, owner_name = ?, owner_name_native = ?,
            seller_name = ?, seller_name_native = ?, buyer_name = ?, buyer_name_native = ?,
            share_hissa = ?, total_area = ?, land_use = ?, land_use_native = ?,
            boundaries_desc = ?, boundaries_desc_native = ?, consideration_amount = ?,
            registration_date = ?, mutation_date = ?, serial_number = ?,
            document_type = ?, document_type_native = ?, status = ?, validation_flags = ?
        WHERE id = ?
    """, (
        payload.district, payload.district_native,
        payload.tehsil, payload.tehsil_native,
        payload.village, payload.village_native,
        payload.khasra_number, payload.khata_number,
        payload.plot_number, payload.property_id,
        payload.owner_name, payload.owner_name_native,
        payload.seller_name, payload.seller_name_native,
        payload.buyer_name, payload.buyer_name_native,
        payload.share_hissa, payload.total_area,
        payload.land_use, payload.land_use_native,
        payload.boundaries_desc, payload.boundaries_desc_native,
        payload.consideration_amount,
        payload.registration_date, payload.mutation_date, payload.serial_number,
        payload.document_type or "Sale Deed", payload.document_type_native,
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
