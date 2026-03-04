"""
╔══════════════════════════════════════════════════════════════════════╗
║                    DAYMARK — STUDENT OS v2.0                         ║
║     Automation-first academic & productivity OS for students         ║
║   PostgreSQL + FastAPI + Groq AI + Full SPA Frontend                 ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP:
  pip install fastapi uvicorn sqlalchemy psycopg2-binary pypdf python-dotenv openai python-multipart

ENV VARS (create a .env file or set in environment):
  DATABASE_URL=postgresql://user:password@localhost:5432/daymark
  GROQ_API_KEY=your_groq_api_key_here
  GROQ_MODEL=llama-3.3-70b-versatile   (or llama3-8b-8192, mixtral-8x7b-32768)
  SECRET_KEY=your_random_secret_here   (optional, for future auth)

RUN:
  python daymark_app.py
  OR
  uvicorn daymark_app:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
Get your free Groq API key at: https://console.groq.com
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os, io, re, json
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, Text, DateTime,
    Boolean, ForeignKey, Float, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from openai import AsyncOpenAI as GroqAsyncOpenAI
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./daymark.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # or llama3-8b-8192, mixtral-8x7b-32768, gemma2-9b-it

# Initialise Groq client via OpenAI-compatible SDK
# Groq uses the OpenAI-compatible SDK with async support — no event loop blocking
_groq_client = None
if GROQ_SDK_AVAILABLE and GROQ_API_KEY:
    _groq_client = GroqAsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════

class Semester(Base):
    __tablename__ = "semesters"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    status = Column(String(20), default='active', nullable=False)
    archived_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    subjects = relationship("Subject", back_populates="semester", cascade="all, delete-orphan")

class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True, index=True)
    semester_id = Column(Integer, ForeignKey("semesters.id"), nullable=False)
    code = Column(String(20), nullable=False)
    name = Column(String(200), nullable=False)
    credits = Column(Integer)
    status = Column(String(20), default='active', nullable=False)
    archived_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    semester = relationship("Semester", back_populates="subjects")
    course_outcomes = relationship("CourseOutcome", back_populates="subject", cascade="all, delete-orphan")
    resources = relationship("Resource", back_populates="subject", cascade="all, delete-orphan")

class CourseOutcome(Base):
    __tablename__ = "course_outcomes"
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    co_code = Column(String(20), nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text)
    status = Column(String(20), default='active', nullable=False)
    archived_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    subject = relationship("Subject", back_populates="course_outcomes")

class Resource(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    title = Column(String(200), nullable=False)
    resource_type = Column(String(20), nullable=False)
    url = Column(Text)
    file_path = Column(Text)
    raw_text = Column(Text)
    status = Column(String(20), default='active', nullable=False)
    archived_at = Column(DateTime, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    subject = relationship("Subject", back_populates="resources")

class DayLog(Base):
    __tablename__ = "day_logs"
    id = Column(Integer, primary_key=True, index=True)
    log_date = Column(Date, nullable=False, unique=True, index=True)
    locked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    locked_at = Column(DateTime, nullable=True)
    entries = relationship("Entry", back_populates="day_log", cascade="all, delete-orphan")

class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True, index=True)
    day_log_id = Column(Integer, ForeignKey("day_logs.id", ondelete="CASCADE"), nullable=False)
    entry_type = Column(String(10), nullable=False)
    text = Column(Text, nullable=False)
    done = Column(Boolean, default=False)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True)
    co_id = Column(Integer, ForeignKey("course_outcomes.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    day_log = relationship("DayLog", back_populates="entries")

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='active')
    priority = Column(String(10), default='medium')
    deadline = Column(Date, nullable=True)
    progress = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    tasks = relationship("ProjectTask", back_populates="project", cascade="all, delete-orphan")

class ProjectTask(Base):
    __tablename__ = "project_tasks"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    text = Column(Text, nullable=False)
    done = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    project = relationship("Project", back_populates="tasks")

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    tags = Column(String(500))
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class AIChat(Base):
    __tablename__ = "ai_chats"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    context = Column(String(50), default='general')
    created_at = Column(DateTime, default=datetime.utcnow)

class TimetableSlot(Base):
    __tablename__ = "timetable_slots"
    id = Column(Integer, primary_key=True, index=True)
    day_of_week = Column(Integer, nullable=False)   # 0=Monday … 6=Sunday
    time_start   = Column(String(8), nullable=False) # "09:00"
    time_end     = Column(String(8), nullable=False) # "10:00"
    subject_name = Column(String(200), nullable=False)
    subject_code = Column(String(30))
    room         = Column(String(100))
    teacher      = Column(String(200))
    notes        = Column(Text)
    subject_id   = Column(Integer, ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

# ══════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════

class SemesterCreate(BaseModel):
    name: str
    start_date: date
    end_date: date

class SemesterOut(SemesterCreate):
    id: int
    status: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class SubjectCreate(BaseModel):
    semester_id: int
    code: str
    name: str
    credits: Optional[int] = None

class SubjectOut(SubjectCreate):
    id: int
    status: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class CourseOutcomeCreate(BaseModel):
    subject_id: int
    co_code: str
    title: str
    description: Optional[str] = None

class CourseOutcomeOut(CourseOutcomeCreate):
    id: int
    status: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class EntryCreate(BaseModel):
    entry_type: str
    text: str
    done: bool = False
    subject_id: Optional[int] = None
    co_id: Optional[int] = None

class EntryOut(EntryCreate):
    id: int
    day_log_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class DayLogSync(BaseModel):
    habits: List[EntryCreate]
    tasks: List[EntryCreate]

class DayLogResponse(BaseModel):
    id: Optional[int]
    log_date: date
    locked: bool
    habits: List[EntryOut]
    tasks: List[EntryOut]
    created_at: Optional[datetime]
    locked_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    priority: str = 'medium'
    deadline: Optional[date] = None

class ProjectTaskCreate(BaseModel):
    text: str

class NoteCreate(BaseModel):
    title: str
    content: Optional[str] = None
    tags: Optional[str] = None
    subject_id: Optional[int] = None

class AIChatMessage(BaseModel):
    message: str
    context: str = 'general'

class TimetableSlotCreate(BaseModel):
    day_of_week: int
    time_start: str
    time_end: str
    subject_name: str
    subject_code: Optional[str] = None
    room: Optional[str] = None
    teacher: Optional[str] = None
    notes: Optional[str] = None
    subject_id: Optional[int] = None

class TimetableSlotOut(TimetableSlotCreate):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# ══════════════════════════════════════════════════════════════════════
# AI SERVICES  (Groq via OpenAI-compatible SDK)
# ══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# PDF TEXT CLEANING — normalise raw extracted text
# ──────────────────────────────────────────────────────────
def _clean_pdf_text(raw: str) -> str:
    """Fix common PDF extraction artifacts so patterns/AI both work better."""
    t = raw
    # Re-join hyphenated line-breaks  e.g.  "Oper-\nating" → "Operating"
    t = re.sub(r'-\n', '', t)
    # Collapse multiple blank lines to double newline
    t = re.sub(r'\n{3,}', '\n\n', t)
    # Remove page numbers / headers that are lone numbers on a line
    t = re.sub(r'(?m)^\s*\d+\s*$', '', t)
    # Strip running headers like "RCCIT – CS501 – Operating Systems"
    t = re.sub(r'(?i)^.{0,60}(syllabus|curriculum|rcc|university|college|department).{0,60}$',
               '', t, flags=re.MULTILINE)
    # Normalise unicode dashes/bullets
    t = t.replace('\u2013', '-').replace('\u2014', '-').replace('\u2022', '*')
    return t.strip()

# ──────────────────────────────────────────────────────────
# MULTI-STRATEGY CO EXTRACTOR (no AI required)
# ──────────────────────────────────────────────────────────
def _extract_cos_pattern(pdf_text: str, subject_name: str) -> Dict:
    text = _clean_pdf_text(pdf_text)
    if not text or len(text) < 50:
        return {"success": False, "error": "PDF too short or unreadable",
                "course_outcomes": [], "warnings": ["Could not extract text"], "confidence": "none"}

    cos: list = []
    method = "pattern"

    # ── Strategy 1: explicit "CO Section" block ──────────────
    # Looks for a section header like "Course Outcomes", "CO:", "Learning Outcomes"
    # then grabs numbered items inside that section
    section_match = re.search(
        r'(?i)(course\s+outcomes?|learning\s+outcomes?|co\s*:)\s*\n(.*?)(?=\n[A-Z][A-Z\s]{5,}\n|\Z)',
        text, re.DOTALL)
    if section_match:
        block = section_match.group(2)
        # numbered items: "1. ...", "CO1:", "CO 1 –", "1)" etc.
        items = re.findall(
            r'(?:CO\s*[-:]?\s*)?(\d+)[.:\)\-–]\s*(.+?)(?=(?:CO\s*[-:]?\s*)?\d+[.:\)\-–]|$)',
            block, re.DOTALL | re.IGNORECASE)
        for num, body in items:
            body = re.sub(r'\s+', ' ', body).strip()
            body = re.sub(r'\(K\d+\)', '', body).strip()
            body = re.sub(r'Students will (?:be )?able to[:\s\.]+', '', body, flags=re.IGNORECASE).strip()
            if len(body) > 8:
                cos.append({"co_code": f"CO{num}", "title": body[:300], "description": ""})

    # ── Strategy 2: inline CO labels anywhere in document ────
    if not cos:
        patterns = [
            r'CO\s*[-:]?\s*(\d+)\s*[:\-\.–]\s*(.+?)(?=CO\s*[-:]?\s*\d+|\n\n|$)',
            r'(\d+)\.\s+((?:Students? (?:will|shall)|Upon completion|After studying|Able to).+?)(?=\d+\.|\n\n|$)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE | re.DOTALL):
                body = re.sub(r'\s+', ' ', m.group(2)).strip()[:350]
                if len(body) > 8:
                    cos.append({"co_code": f"CO{m.group(1)}", "title": body, "description": ""})
            if cos:
                break

    # ── Strategy 3: RCCIIT-specific pattern ──────────────────
    if not cos:
        for m in re.finditer(r'(?:RCC-[A-Z\-]+\.)?CO(\d+)\s+(.+?)(?=(?:RCC-[A-Z\-]+\.)?CO\d+|\Z)',
                             text, re.DOTALL):
            body = re.sub(r'\s+', ' ', m.group(2)).strip()
            body = re.sub(r'\(K\d+\)|Students will (?:be )?able to[:\s.]+', '', body, flags=re.IGNORECASE).strip()
            if len(body) > 8:
                cos.append({"co_code": f"CO{m.group(1)}", "title": body[:300], "description": ""})

    # ── Strategy 4: bullet / star list after outcome keywords ─
    if not cos:
        for m in re.finditer(r'(?i)(?:outcome|objective)s?[:\n](.+?)(?=\n[A-Z]{3}|\Z)',
                             text, re.DOTALL):
            bullets = re.findall(r'[*\-•►▪\d]+[.)\s]+(.+?)(?=[*\-•►▪\d]+[.)\s]|$)',
                                 m.group(1), re.DOTALL)
            for i, b in enumerate(bullets, 1):
                b = re.sub(r'\s+', ' ', b).strip()[:300]
                if len(b) > 8:
                    cos.append({"co_code": f"CO{i}", "title": b, "description": ""})
            if cos:
                break

    if cos:
        # Deduplicate by co_code, keep first occurrence
        seen, unique = set(), []
        for c in cos:
            if c["co_code"] not in seen:
                seen.add(c["co_code"]); unique.append(c)
        cos = unique
        try:
            cos.sort(key=lambda x: int(re.search(r'\d+', x["co_code"]).group()))
        except Exception:
            pass
        return {"success": True, "course_outcomes": cos, "confidence": "medium",
                "warnings": [f"Found {len(cos)} COs via pattern matching"], "method": method}

    return {"success": True, "course_outcomes": [], "confidence": "none",
            "warnings": ["No course outcomes detected. Edit them manually below."], "method": "none"}


# ──────────────────────────────────────────────────────────
# SUBJECT INFO EXTRACTOR (no AI required)
# ──────────────────────────────────────────────────────────
def _extract_subject_info(text: str) -> Dict:
    """Try to pull subject name, code and credits from the first 2000 chars."""
    header = _clean_pdf_text(text[:2000])

    # Subject code — patterns like CS501, BCA-301, IT-3101, 20CS501
    code_match = re.search(r'\b([A-Z]{2,5}[-–]?\d{3,6})\b', header)
    code = code_match.group(1) if code_match else ""

    # Credits — "4 Credits", "Credit: 4", "L-T-P: 3-1-0" (sum)
    credits = None
    cr_match = re.search(r'(\d)\s*(?:credit|cr\.)', header, re.IGNORECASE)
    if cr_match:
        credits = int(cr_match.group(1))
    else:
        ltp = re.search(r'L[\s\-]+T[\s\-]+P[:\s]*(\d+)[\s\-]+(\d+)[\s\-]+(\d+)', header, re.IGNORECASE)
        if ltp:
            credits = int(ltp.group(1)) + int(ltp.group(2))

    # Subject name — line that contains the code or has "Subject:" / is title-cased 3+ words
    name = ""
    for line in header.split("\n"):
        line = line.strip()
        if code and code in line:
            cleaned = re.sub(re.escape(code), '', line).strip('–-:| ').strip()
            if len(cleaned) > 4:
                name = cleaned[:120]; break
    if not name:
        subj_match = re.search(r'(?i)subject\s*(?:name|title)?\s*[:\-]?\s*(.+)', header)
        if subj_match:
            name = subj_match.group(1).strip()[:120]
    if not name:
        # First long title-case line
        for line in header.split("\n"):
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 8 and sum(1 for w in words if w[0:1].isupper()) >= 2:
                name = line[:120]; break

    return {"name": name or "Unknown Subject", "code": code or "SUBJ", "credits": credits}


# ──────────────────────────────────────────────────────────
# TIMETABLE PARSER (no AI required)
# ──────────────────────────────────────────────────────────
def _parse_timetable_pattern(text: str) -> List[Dict]:
    """Heuristic timetable parser — works on most tabular PDF schedules."""
    text = _clean_pdf_text(text)
    DAY_MAP = {
        "monday":0,"mon":0,"mo":0,
        "tuesday":1,"tue":1,"tu":1,
        "wednesday":2,"wed":2,"we":2,
        "thursday":3,"thu":3,"th":3,
        "friday":4,"fri":4,"fr":4,
        "saturday":5,"sat":5,"sa":5,
        "sunday":6,"sun":6,"su":6,
    }
    TIME_RE = re.compile(r'(\d{1,2})[:\.]?(\d{2})\s*(?:–|-|to)\s*(\d{1,2})[:\.]?(\d{2})')
    TIME_SINGLE = re.compile(r'(\d{1,2})[:\.]?(\d{2})\s*(?:am|pm)?', re.IGNORECASE)

    slots = []
    lines = text.split("\n")

    current_day = None
    for line in lines:
        stripped = line.strip().lower()

        # Detect day headers
        for day_name, day_idx in DAY_MAP.items():
            if re.search(r'\b' + day_name + r'\b', stripped):
                current_day = day_idx
                break

        if current_day is None:
            continue

        # Look for time ranges on this line
        range_match = TIME_RE.search(line)
        if range_match:
            h1,m1,h2,m2 = range_match.groups()
            t_start = f"{int(h1):02d}:{int(m1):02d}"
            t_end   = f"{int(h2):02d}:{int(m2):02d}"
            # Subject name = everything before/after the time, cleaned
            subj_part = TIME_RE.sub('', line).strip()
            subj_part = re.sub(r'[|\t]+', ' ', subj_part).strip()
            # Grab first capitalised-looking chunk as subject name
            subj_match = re.search(r'([A-Z][A-Za-z0-9& ]{3,50})', subj_part)
            subj_name = subj_match.group(1).strip() if subj_match else subj_part[:50]
            # Room number
            room_match = re.search(r'(?:room|hall|lab|rm\.?)\s*[-:]?\s*([\w\d]+)', line, re.IGNORECASE)
            room = room_match.group(1) if room_match else ""
            # Code
            code_match = re.search(r'\b([A-Z]{2,5}\d{3,6})\b', line)
            code = code_match.group(1) if code_match else ""
            if subj_name and len(subj_name) > 2:
                slots.append({
                    "day_of_week": current_day,
                    "time_start": t_start,
                    "time_end": t_end,
                    "subject_name": subj_name,
                    "subject_code": code,
                    "room": room,
                    "teacher": "",
                })

    # Deduplicate (same day+start+subject)
    seen, unique = set(), []
    for s in slots:
        key = (s["day_of_week"], s["time_start"], s["subject_name"].lower())
        if key not in seen:
            seen.add(key); unique.append(s)
    return unique


# ──────────────────────────────────────────────────────────
# AI-ASSISTED EXTRACTORS (call pattern first, AI to fill gaps)
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
# PDF TEXT CLEANING
# ──────────────────────────────────────────────────────────
def _clean_pdf_text(raw: str) -> str:
    t = raw
    t = re.sub(r'-\n', '', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'(?m)^\s*\d+\s*$', '', t)
    t = t.replace('\u2013', '-').replace('\u2014', '-').replace('\u2022', '*')
    return t.strip()

def _extract_cos_pattern(pdf_text: str, subject_name: str) -> Dict:
    text = _clean_pdf_text(pdf_text)
    if not text or len(text) < 50:
        return {"success": False, "error": "PDF too short",
                "course_outcomes": [], "warnings": ["Could not extract text"], "confidence": "none"}
    cos = []
    # Strategy 1: CO section block
    sec = re.search(
        r'(?i)(course\s+outcomes?|learning\s+outcomes?|co\s*:)\s*\n(.*?)(?=\n[A-Z][A-Z\s]{5,}\n|\Z)',
        text, re.DOTALL)
    if sec:
        block = sec.group(2)
        items = re.findall(
            r'(?:CO\s*[-:]?\s*)?(\d+)[.:\)\-\u2013]\s*(.+?)(?=(?:CO\s*[-:]?\s*)?\d+[.:\)\-\u2013]|$)',
            block, re.DOTALL | re.IGNORECASE)
        for num, body in items:
            body = re.sub(r'\s+', ' ', body).strip()
            body = re.sub(r'\(K\d+\)', '', body).strip()
            body = re.sub(r'Students will (?:be )?able to[:\s\.]+', '', body, flags=re.IGNORECASE).strip()
            if len(body) > 8:
                cos.append({"co_code": f"CO{num}", "title": body[:300], "description": ""})
    # Strategy 2: inline CO labels
    if not cos:
        for pat in [
            r'CO\s*[-:]?\s*(\d+)\s*[:\-\.\u2013]\s*(.+?)(?=CO\s*[-:]?\s*\d+|\n\n|$)',
            r'(\d+)\.\s+((?:Students? (?:will|shall)|Upon completion|After studying|Able to).+?)(?=\d+\.|\n\n|$)',
        ]:
            for m in re.finditer(pat, text, re.IGNORECASE | re.DOTALL):
                body = re.sub(r'\s+', ' ', m.group(2)).strip()[:350]
                if len(body) > 8:
                    cos.append({"co_code": f"CO{m.group(1)}", "title": body, "description": ""})
            if cos:
                break
    # Strategy 3: RCCIIT pattern
    if not cos:
        for m in re.finditer(r'(?:RCC-[A-Z\-]+\.)?CO(\d+)\s+(.+?)(?=(?:RCC-[A-Z\-]+\.)?CO\d+|\Z)', text, re.DOTALL):
            body = re.sub(r'\s+', ' ', m.group(2)).strip()
            body = re.sub(r'\(K\d+\)|Students will (?:be )?able to[:\s.]+', '', body, flags=re.IGNORECASE).strip()
            if len(body) > 8:
                cos.append({"co_code": f"CO{m.group(1)}", "title": body[:300], "description": ""})
    # Strategy 4: bullet list after outcome keywords
    if not cos:
        for m in re.finditer(r'(?i)(?:outcome|objective)s?[:\n](.+?)(?=\n[A-Z]{3}|\Z)', text, re.DOTALL):
            bullets = re.findall(r'[*\-\u2022\u25ba\u25aa\d]+[.)\s]+(.+?)(?=[*\-\u2022\u25ba\u25aa\d]+[.)\s]|$)', m.group(1), re.DOTALL)
            for i, b in enumerate(bullets, 1):
                b = re.sub(r'\s+', ' ', b).strip()[:300]
                if len(b) > 8:
                    cos.append({"co_code": f"CO{i}", "title": b, "description": ""})
            if cos:
                break
    if cos:
        seen, unique = set(), []
        for c in cos:
            if c["co_code"] not in seen:
                seen.add(c["co_code"]); unique.append(c)
        try:
            unique.sort(key=lambda x: int(re.search(r'\d+', x["co_code"]).group()))
        except Exception:
            pass
        return {"success": True, "course_outcomes": unique, "confidence": "medium",
                "warnings": [f"Found {len(unique)} COs via pattern matching"], "method": "pattern"}
    return {"success": True, "course_outcomes": [], "confidence": "none",
            "warnings": ["No course outcomes detected. Edit them manually below."], "method": "none"}


def _extract_subject_info(text: str) -> Dict:
    header = _clean_pdf_text(text[:2000])
    code_match = re.search(r'\b([A-Z]{2,5}[-\u2013]?\d{3,6})\b', header)
    code = code_match.group(1) if code_match else ""
    credits = None
    cr = re.search(r'(\d)\s*(?:credit|cr\.)', header, re.IGNORECASE)
    if cr:
        credits = int(cr.group(1))
    else:
        ltp = re.search(r'L[\s\-]+T[\s\-]+P[:\s]*(\d+)[\s\-]+(\d+)[\s\-]+(\d+)', header, re.IGNORECASE)
        if ltp:
            credits = int(ltp.group(1)) + int(ltp.group(2))
    name = ""
    for line in header.split("\n"):
        line = line.strip()
        if code and code in line:
            cleaned = re.sub(re.escape(code), '', line).strip('\u2013-:| ').strip()
            if len(cleaned) > 4:
                name = cleaned[:120]; break
    if not name:
        sm = re.search(r'(?i)subject\s*(?:name|title)?\s*[:\-]?\s*(.+)', header)
        if sm:
            name = sm.group(1).strip()[:120]
    if not name:
        for line in header.split("\n"):
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 8 and sum(1 for w in words if w[:1].isupper()) >= 2:
                name = line[:120]; break
    return {"name": name or "Unknown Subject", "code": code or "SUBJ", "credits": credits}


def _parse_timetable_pattern(text: str) -> List[Dict]:
    text = _clean_pdf_text(text)
    DAY_MAP = {
        "monday":0,"mon":0,"tuesday":1,"tue":1,"wednesday":2,"wed":2,
        "thursday":3,"thu":3,"friday":4,"fri":4,"saturday":5,"sat":5,"sunday":6,"sun":6,
    }
    TIME_RE = re.compile(r'(\d{1,2})[:\.]?(\d{2})\s*(?:\u2013|-|to)\s*(\d{1,2})[:\.]?(\d{2})')
    slots = []
    current_day = None
    for line in text.split("\n"):
        stripped = line.strip().lower()
        for day_name, day_idx in DAY_MAP.items():
            if re.search(r'\b' + day_name + r'\b', stripped):
                current_day = day_idx
                break
        if current_day is None:
            continue
        rm = TIME_RE.search(line)
        if rm:
            h1,m1,h2,m2 = rm.groups()
            t_start = f"{int(h1):02d}:{int(m1):02d}"
            t_end   = f"{int(h2):02d}:{int(m2):02d}"
            subj_part = TIME_RE.sub('', line).strip()
            subj_part = re.sub(r'[|\t]+', ' ', subj_part).strip()
            sm = re.search(r'([A-Z][A-Za-z0-9& ]{3,50})', subj_part)
            subj_name = sm.group(1).strip() if sm else subj_part[:50]
            room_m = re.search(r'(?:room|hall|lab|rm\.?)\s*[-:]?\s*([\w\d]+)', line, re.IGNORECASE)
            room = room_m.group(1) if room_m else ""
            code_m = re.search(r'\b([A-Z]{2,5}\d{3,6})\b', line)
            code = code_m.group(1) if code_m else ""
            if subj_name and len(subj_name) > 2:
                slots.append({"day_of_week":current_day,"time_start":t_start,"time_end":t_end,
                               "subject_name":subj_name,"subject_code":code,"room":room,"teacher":""})
    seen, unique = set(), []
    for s in slots:
        key = (s["day_of_week"], s["time_start"], s["subject_name"].lower())
        if key not in seen:
            seen.add(key); unique.append(s)
    return unique


async def extract_cos_with_groq(pdf_text: str, subject_name: str) -> Dict:
    pattern_result = _extract_cos_pattern(pdf_text, subject_name)
    if pattern_result["course_outcomes"]:
        return {**pattern_result, "warnings": [f"✅ Extracted {len(pattern_result['course_outcomes'])} COs"]}
    if not _groq_client:
        return pattern_result
    prompt = (
        f"Extract ALL Course Outcomes from this syllabus for: {subject_name}\n\n"
        f"TEXT:\n{_clean_pdf_text(pdf_text)[:7000]}\n\n"
        'Return ONLY valid JSON: {"course_outcomes":[{"co_code":"CO1","title":"...","description":"..."}]}'
    )
    try:
        resp = await _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content":"Extract course outcomes. Return only valid JSON."},
                {"role":"user","content":prompt},
            ],
            temperature=0.1,
        )
        raw = re.sub(r'^```(?:json)?\s*', '', resp.choices[0].message.content.strip())
        raw = re.sub(r'\s*```$', '', raw).strip()
        data = json.loads(raw)
        cos = data.get("course_outcomes", [])
        if cos:
            return {"success":True,"course_outcomes":cos,"confidence":"high",
                    "warnings":[f"✅ Groq extracted {len(cos)} COs"],"method":"groq"}
    except Exception as e:
        print(f"Groq CO error: {e}")
    return pattern_result


async def parse_routine_pdf_with_groq(pdf_text: str) -> Dict:
    slots = _parse_timetable_pattern(pdf_text)
    if slots:
        return {"success":True,"slots":slots,"count":len(slots),"method":"pattern"}
    if not _groq_client:
        return {"success":False,"error":"Could not parse timetable. Ensure the PDF has selectable text.","slots":[]}
    prompt = (
        "Parse this college timetable PDF. Return ONLY valid JSON:\n"
        '{"slots":[{"day_of_week":0,"time_start":"09:00","time_end":"10:00","subject_name":"OS","subject_code":"CS501","room":"301","teacher":""}]}\n\n'
        "day_of_week: 0=Monday...6=Sunday. Time in 24h HH:MM.\n\n"
        f"TIMETABLE:\n{_clean_pdf_text(pdf_text)[:9000]}"
    )
    try:
        resp = await _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content":"Extract timetable data. Return only valid JSON."},
                {"role":"user","content":prompt},
            ],
            temperature=0.05,
        )
        raw = re.sub(r'^```(?:json)?\s*', '', resp.choices[0].message.content.strip())
        raw = re.sub(r'\s*```$', '', raw).strip()
        data = json.loads(raw)
        slots = data.get("slots", [])
        for s in slots:
            s["day_of_week"] = max(0, min(6, int(s.get("day_of_week", 0))))
        return {"success":True,"slots":slots,"count":len(slots),"method":"groq"}
    except Exception as e:
        print(f"Routine parse error: {e}")
        return {"success":False,"error":"Could not parse timetable.","slots":[]}


async def parse_subject_pdf_with_groq(pdf_text: str) -> Dict:
    subj_info = _extract_subject_info(pdf_text)
    cos_result = _extract_cos_pattern(pdf_text, subj_info.get("name",""))
    cos = cos_result.get("course_outcomes", [])
    if subj_info.get("name","Unknown Subject") != "Unknown Subject" and cos:
        return {"success":True,"subject":subj_info,"course_outcomes":cos,"method":"pattern"}
    if _groq_client:
        prompt = (
            "Parse this college syllabus PDF. Return ONLY valid JSON:\n"
            '{"subject":{"name":"Operating Systems","code":"CS501","credits":4},'
            '"course_outcomes":[{"co_code":"CO1","title":"short title","description":"full text"}]}\n\n'
            f"SYLLABUS:\n{_clean_pdf_text(pdf_text)[:9000]}"
        )
        try:
            resp = await _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role":"system","content":"Extract academic data from syllabus PDFs. Return only valid JSON."},
                    {"role":"user","content":prompt},
                ],
                temperature=0.1,
            )
            raw = re.sub(r'^```(?:json)?\s*', '', resp.choices[0].message.content.strip())
            raw = re.sub(r'\s*```$', '', raw).strip()
            data = json.loads(raw)
            ai_s = data.get("subject", {})
            ai_cos = data.get("course_outcomes", [])
            final_subj = {
                "name":    ai_s.get("name")    or subj_info.get("name","Unknown Subject"),
                "code":    ai_s.get("code")    or subj_info.get("code","SUBJ"),
                "credits": ai_s.get("credits") or subj_info.get("credits"),
            }
            final_cos = ai_cos if len(ai_cos) >= len(cos) else cos
            return {"success":True,"subject":final_subj,"course_outcomes":final_cos,"method":"groq"}
        except Exception as e:
            print(f"Subject AI error: {e}")
    if cos or subj_info.get("code","SUBJ") != "SUBJ":
        return {"success":True,"subject":subj_info,"course_outcomes":cos,"method":"pattern_only"}
    return {"success":False,"error":"Could not extract info. Ensure PDF has selectable text.",
            "subject":None,"course_outcomes":[]}



async def ask_groq(message: str, context: str, db: Session) -> str:
    """Send a message to Groq with full app context — ultra-fast inference."""
    if not _groq_client:
        return "⚠️ Groq AI not configured. Add GROQ_API_KEY to your .env file. Get a free key at https://console.groq.com"

    # Build a brief context string from live DB data
    ctx_parts: List[str] = []
    today = date.today()

    day_log = db.query(DayLog).filter(DayLog.log_date == today).first()
    if day_log:
        entries = db.query(Entry).filter(Entry.day_log_id == day_log.id).all()
        done = sum(1 for e in entries if e.done)
        ctx_parts.append(
            f"Today ({today}): {done}/{len(entries)} tasks done, "
            f"day {'locked' if day_log.locked else 'active'}"
        )

    semesters = db.query(Semester).filter(Semester.status == 'active').all()
    for sem in semesters:
        subjects = db.query(Subject).filter(
            Subject.semester_id == sem.id, Subject.status == 'active'
        ).all()
        ctx_parts.append(f"Semester: {sem.name} ({len(subjects)} active subjects)")

    projects = db.query(Project).filter(Project.status == 'active').all()
    if projects:
        ctx_parts.append(f"Active projects: {', '.join(p.name for p in projects[:5])}")

    # Fetch last 6 messages for conversation history
    recent_chats = db.query(AIChat).order_by(AIChat.created_at.desc()).limit(6).all()
    history = [
        {"role": "user" if c.role == "user" else "assistant", "content": c.content}
        for c in reversed(recent_chats)
    ]

    # Add today's timetable to context
    dow = today.weekday()
    today_slots = db.query(TimetableSlot).filter(TimetableSlot.day_of_week == dow).order_by(TimetableSlot.time_start).all()
    if today_slots:
        slot_strs = [f"{s.time_start}-{s.time_end} {s.subject_name}{(' @'+s.room) if s.room else ''}" for s in today_slots]
        ctx_parts.append(f"Today's classes: {'; '.join(slot_strs)}")

    projects = db.query(Project).filter(Project.status == 'active').all()
    if projects:
        ctx_parts.append(f"Active projects: {', '.join(p.name for p in projects[:5])}")

    system_prompt = (
        f"You are DayMark AI — an intelligent academic assistant for a college student.\n"
        f"You help with study planning, exam preparation, timetable management, and productivity.\n"
        f"Give specific, actionable advice. Use bullet points. Be direct and practical.\n"
        f"Context: {'; '.join(ctx_parts) or 'No data yet'}\n"
        f"Today: {today.strftime('%A, %B %d, %Y')}"
    )

    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        response = await _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Groq error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("✅ DayMark database ready")
    yield

app = FastAPI(title="DayMark Student OS", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — SEMESTERS
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/semesters", response_model=SemesterOut)
def create_semester(s: SemesterCreate, db: Session = Depends(get_db)):
    obj = Semester(**s.dict())
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@app.get("/api/semesters", response_model=List[SemesterOut])
def list_semesters(status: str = Query('active'), db: Session = Depends(get_db)):
    q = db.query(Semester)
    if status != 'all': q = q.filter(Semester.status == status)
    return q.order_by(Semester.created_at.desc()).all()

@app.patch("/api/semesters/{sid}")
def update_semester(sid: int, data: dict, db: Session = Depends(get_db)):
    sem = db.query(Semester).filter(Semester.id == sid).first()
    if not sem: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(sem, k): setattr(sem, k, v)
    db.commit(); return {"ok": True}

@app.delete("/api/semesters/{sid}")
def delete_semester(sid: int, mode: str = Query('soft'), db: Session = Depends(get_db)):
    sem = db.query(Semester).filter(Semester.id == sid).first()
    if not sem: raise HTTPException(404, "Not found")
    if mode == 'hard':
        db.delete(sem)
    elif mode == 'archive':
        sem.status = 'archived'; sem.archived_at = datetime.utcnow()
    else:
        sem.status = 'hidden'
    db.commit()
    return {"ok": True, "mode": mode}

@app.post("/api/semesters/{sid}/restore")
def restore_semester(sid: int, db: Session = Depends(get_db)):
    sem = db.query(Semester).filter(Semester.id == sid).first()
    if not sem: raise HTTPException(404, "Not found")
    sem.status = 'active'; sem.archived_at = None
    for sub in sem.subjects:
        sub.status = 'active'; sub.archived_at = None
    db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — SUBJECTS
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/subjects", response_model=SubjectOut)
def create_subject(s: SubjectCreate, db: Session = Depends(get_db)):
    obj = Subject(**s.dict())
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@app.get("/api/semesters/{sid}/subjects", response_model=List[SubjectOut])
def list_subjects(sid: int, status: str = Query('active'), db: Session = Depends(get_db)):
    q = db.query(Subject).filter(Subject.semester_id == sid)
    if status != 'all': q = q.filter(Subject.status == status)
    return q.all()

@app.get("/api/subjects/all", response_model=List[SubjectOut])
def list_all_subjects(db: Session = Depends(get_db)):
    return db.query(Subject).filter(Subject.status == 'active').all()

@app.patch("/api/subjects/{sid}")
def update_subject(sid: int, data: dict, db: Session = Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == sid).first()
    if not sub: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(sub, k) and k not in ('id','semester_id','created_at'): setattr(sub, k, v)
    db.commit(); return {"ok": True}

@app.delete("/api/subjects/{sid}")
def delete_subject(sid: int, mode: str = Query('soft'), db: Session = Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == sid).first()
    if not sub: raise HTTPException(404, "Not found")
    if mode == 'hard': db.delete(sub)
    elif mode == 'archive': sub.status = 'archived'; sub.archived_at = datetime.utcnow()
    else: sub.status = 'hidden'
    db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — COURSE OUTCOMES
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/course-outcomes", response_model=CourseOutcomeOut)
def create_co(co: CourseOutcomeCreate, db: Session = Depends(get_db)):
    obj = CourseOutcome(**co.dict())
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@app.post("/api/course-outcomes/bulk")
def create_cos_bulk(cos: List[CourseOutcomeCreate], db: Session = Depends(get_db)):
    created = []
    for co in cos:
        obj = CourseOutcome(**co.dict())
        db.add(obj); db.flush(); created.append({"id": obj.id, "co_code": obj.co_code})
    db.commit()
    return {"created": len(created), "items": created}

@app.get("/api/subjects/{sid}/course-outcomes", response_model=List[CourseOutcomeOut])
def list_cos(sid: int, status: str = Query('active'), db: Session = Depends(get_db)):
    q = db.query(CourseOutcome).filter(CourseOutcome.subject_id == sid)
    if status != 'all': q = q.filter(CourseOutcome.status == status)
    return q.all()

@app.patch("/api/course-outcomes/{cid}")
def update_co(cid: int, data: dict, db: Session = Depends(get_db)):
    co = db.query(CourseOutcome).filter(CourseOutcome.id == cid).first()
    if not co: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(co, k) and k not in ('id','subject_id','created_at'): setattr(co, k, v)
    db.commit(); return {"ok": True}

@app.delete("/api/course-outcomes/{cid}")
def delete_co(cid: int, mode: str = Query('soft'), db: Session = Depends(get_db)):
    co = db.query(CourseOutcome).filter(CourseOutcome.id == cid).first()
    if not co: raise HTTPException(404, "Not found")
    entry_count = db.query(Entry).filter(Entry.co_id == cid).count()
    if mode == 'hard' and entry_count > 0:
        raise HTTPException(400, f"Cannot hard delete CO with {entry_count} linked entries. Archive instead.")
    if mode == 'hard': db.delete(co)
    elif mode == 'archive': co.status = 'archived'; co.archived_at = datetime.utcnow()
    else: co.status = 'hidden'
    db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — RESOURCES / PDF
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/resources/upload")
async def upload_pdf(subject_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject: raise HTTPException(404, "Subject not found")
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0: raise HTTPException(400, "Empty file")
    raw_text = ""
    if PDF_AVAILABLE:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t: raw_text += t + "\n"
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
    if not raw_text or len(raw_text.strip()) < 50:
        raise HTTPException(400, "Could not extract text. PDF may be image-based.")
    resource = Resource(subject_id=subject_id, title=file.filename, resource_type="pdf",
                        file_path=f"/uploads/{file.filename}", raw_text=raw_text)
    db.add(resource); db.commit(); db.refresh(resource)
    # Use Groq AI for CO extraction
    result = await extract_cos_with_groq(raw_text, f"{subject.code} - {subject.name}")
    return {"resource_id": resource.id, "filename": file.filename,
            "text_length": len(raw_text), "raw_text_preview": raw_text[:500],
            "pages_count": len(reader.pages) if PDF_AVAILABLE else 0, "ai_extraction": result}

@app.get("/api/resources/subject/{subject_id}")
def list_resources(subject_id: int, db: Session = Depends(get_db)):
    return db.query(Resource).filter(Resource.subject_id == subject_id, Resource.status == 'active').all()

@app.patch("/api/resources/{rid}")
def update_resource(rid: int, data: dict, db: Session = Depends(get_db)):
    r = db.query(Resource).filter(Resource.id == rid).first()
    if not r: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(r, k) and k not in ('id','subject_id','created_at','uploaded_at'): setattr(r, k, v)
    db.commit(); return {"ok": True}

@app.delete("/api/resources/{rid}")
def delete_resource(rid: int, db: Session = Depends(get_db)):
    r = db.query(Resource).filter(Resource.id == rid).first()
    if not r: raise HTTPException(404, "Not found")
    db.delete(r); db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — TIMETABLE
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/timetable")
def get_timetable(db: Session = Depends(get_db)):
    slots = db.query(TimetableSlot).order_by(TimetableSlot.day_of_week, TimetableSlot.time_start).all()
    return [{"id":s.id,"day_of_week":s.day_of_week,"time_start":s.time_start,"time_end":s.time_end,
             "subject_name":s.subject_name,"subject_code":s.subject_code,"room":s.room,
             "teacher":s.teacher,"notes":s.notes,"subject_id":s.subject_id} for s in slots]

@app.get("/api/timetable/today")
def get_today_timetable(db: Session = Depends(get_db)):
    dow = date.today().weekday()
    slots = db.query(TimetableSlot).filter(TimetableSlot.day_of_week == dow).order_by(TimetableSlot.time_start).all()
    return [{"id":s.id,"time_start":s.time_start,"time_end":s.time_end,
             "subject_name":s.subject_name,"subject_code":s.subject_code,
             "room":s.room,"teacher":s.teacher,"subject_id":s.subject_id} for s in slots]

@app.post("/api/timetable/slots", response_model=TimetableSlotOut)
def create_timetable_slot(data: TimetableSlotCreate, db: Session = Depends(get_db)):
    slot = TimetableSlot(**data.dict())
    db.add(slot); db.commit(); db.refresh(slot)
    return slot

@app.patch("/api/timetable/slots/{sid}")
def update_timetable_slot(sid: int, data: dict, db: Session = Depends(get_db)):
    slot = db.query(TimetableSlot).filter(TimetableSlot.id == sid).first()
    if not slot: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(slot, k): setattr(slot, k, v)
    db.commit(); return {"ok": True}

@app.delete("/api/timetable/slots/{sid}")
def delete_timetable_slot(sid: int, db: Session = Depends(get_db)):
    slot = db.query(TimetableSlot).filter(TimetableSlot.id == sid).first()
    if not slot: raise HTTPException(404, "Not found")
    db.delete(slot); db.commit(); return {"ok": True}

@app.post("/api/timetable/import-pdf")
async def import_timetable_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a routine/timetable PDF and auto-create the full weekly schedule."""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    pdf_bytes = await file.read()
    if not pdf_bytes: raise HTTPException(400, "Empty file")
    raw_text = ""
    if PDF_AVAILABLE:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t: raw_text += t + "\n"
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
    if not raw_text or len(raw_text.strip()) < 30:
        raise HTTPException(400, "Could not extract text from PDF.")
    result = await parse_routine_pdf_with_groq(raw_text)
    if not result["success"] or not result["slots"]:
        return {"imported": 0, "message": "Could not parse timetable. Try a text-based PDF.", "slots": []}
    # Try to auto-link subjects
    all_subjects = db.query(Subject).filter(Subject.status == 'active').all()
    sub_map = {s.name.lower(): s.id for s in all_subjects}
    sub_code_map = {s.code.lower(): s.id for s in all_subjects}
    created = []
    for slot_data in result["slots"]:
        sub_id = (sub_map.get(slot_data["subject_name"].lower()) or
                  sub_code_map.get((slot_data.get("subject_code") or "").lower()))
        slot = TimetableSlot(
            day_of_week=slot_data["day_of_week"],
            time_start=slot_data["time_start"],
            time_end=slot_data["time_end"],
            subject_name=slot_data["subject_name"],
            subject_code=slot_data.get("subject_code", ""),
            room=slot_data.get("room", ""),
            teacher=slot_data.get("teacher", ""),
            subject_id=sub_id,
        )
        db.add(slot); db.flush()
        created.append({"id": slot.id, "day": slot_data["day_of_week"],
                        "subject": slot_data["subject_name"],
                        "time": f"{slot_data['time_start']}-{slot_data['time_end']}"})
    db.commit()
    return {"imported": len(created), "message": f"Successfully imported {len(created)} class slots!", "slots": created}

@app.delete("/api/timetable/clear")
def clear_timetable(db: Session = Depends(get_db)):
    db.query(TimetableSlot).delete()
    db.commit(); return {"ok": True}

@app.post("/api/day/{log_date}/generate-from-timetable")
def generate_day_from_timetable(log_date: date, db: Session = Depends(get_db)):
    """Auto-populate today's tasks directly from the timetable."""
    dow = log_date.weekday()
    slots = db.query(TimetableSlot).filter(TimetableSlot.day_of_week == dow).order_by(TimetableSlot.time_start).all()
    if not slots:
        return {"status": "no_slots", "message": "No classes scheduled for this day in your timetable."}
    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    if not day:
        day = DayLog(log_date=log_date, locked=False)
        db.add(day); db.flush()
    if day.locked:
        raise HTTPException(400, "Day is locked")
    added = 0
    for slot in slots:
        room_str = f" ({slot.room})" if slot.room else ""
        task_text = f"📚 {slot.time_start}-{slot.time_end} {slot.subject_name}{room_str}"
        # Check not already added
        existing = db.query(Entry).filter(Entry.day_log_id == day.id, Entry.text == task_text).first()
        if not existing:
            db.add(Entry(day_log_id=day.id, entry_type="task", text=task_text,
                         done=False, subject_id=slot.subject_id))
            added += 1
    db.commit()
    return {"status": "generated", "added": added, "message": f"Added {added} classes to today's schedule."}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — SUBJECT PDF AUTO-IMPORT
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/subjects/import-from-pdf")
async def import_subject_from_pdf(semester_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a syllabus PDF → auto-create subject + all COs in one step."""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    sem = db.query(Semester).filter(Semester.id == semester_id).first()
    if not sem: raise HTTPException(404, "Semester not found")
    pdf_bytes = await file.read()
    if not pdf_bytes: raise HTTPException(400, "Empty file")
    raw_text = ""
    if PDF_AVAILABLE:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t: raw_text += t + "\n"
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
    if not raw_text or len(raw_text.strip()) < 50:
        raise HTTPException(400, "Could not extract text from PDF.")
    result = await parse_subject_pdf_with_groq(raw_text)
    if not result["success"] or not result.get("subject"):
        return {"success": False, "message": "Could not parse subject info. Try a text-based syllabus PDF.", "subject_id": None}
    sub_info = result["subject"]
    sub = Subject(
        semester_id=semester_id,
        code=sub_info.get("code", "SUBJ"),
        name=sub_info.get("name", file.filename),
        credits=sub_info.get("credits"),
    )
    db.add(sub); db.flush()
    resource = Resource(subject_id=sub.id, title=file.filename,
                        resource_type="pdf", raw_text=raw_text)
    db.add(resource)
    cos_created = 0
    for co_data in result.get("course_outcomes", []):
        co = CourseOutcome(
            subject_id=sub.id,
            co_code=co_data.get("co_code", f"CO{cos_created+1}"),
            title=co_data.get("title", "")[:400],
            description=co_data.get("description", ""),
        )
        db.add(co); cos_created += 1
    db.commit()
    return {
        "success": True,
        "subject_id": sub.id,
        "subject_name": sub.name,
        "subject_code": sub.code,
        "credits": sub.credits,
        "cos_created": cos_created,
        "message": f"✅ Imported '{sub.name}' with {cos_created} course outcomes!",
    }

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — DAILY LOG
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/day/{log_date}", response_model=DayLogResponse)
def get_day(log_date: date, db: Session = Depends(get_db)):
    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    if not day:
        return DayLogResponse(id=None, log_date=log_date, locked=False, habits=[], tasks=[], created_at=None)
    habits = db.query(Entry).filter(Entry.day_log_id == day.id, Entry.entry_type == "habit").all()
    tasks = db.query(Entry).filter(Entry.day_log_id == day.id, Entry.entry_type == "task").all()
    return DayLogResponse(id=day.id, log_date=day.log_date, locked=day.locked,
                          habits=habits, tasks=tasks, created_at=day.created_at, locked_at=day.locked_at)

@app.post("/api/day/{log_date}/sync")
def sync_day(log_date: date, payload: DayLogSync, force: bool = Query(False), db: Session = Depends(get_db)):
    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    if not day:
        day = DayLog(log_date=log_date, locked=False)
        db.add(day); db.flush()
    if day.locked: raise HTTPException(400, "Day is locked")
    existing = db.query(Entry).filter(Entry.day_log_id == day.id).count()
    if not force and existing > 0 and not payload.habits and not payload.tasks:
        return {"status": "ignored_unsafe_empty_sync"}
    db.query(Entry).filter(Entry.day_log_id == day.id).delete(synchronize_session=False)
    for h in payload.habits:
        db.add(Entry(day_log_id=day.id, entry_type="habit", text=h.text, done=h.done,
                     subject_id=h.subject_id, co_id=h.co_id))
    for t in payload.tasks:
        db.add(Entry(day_log_id=day.id, entry_type="task", text=t.text, done=t.done,
                     subject_id=t.subject_id, co_id=t.co_id))
    db.commit(); return {"status": "synced"}

@app.post("/api/day/{log_date}/end")
def end_day(log_date: date, db: Session = Depends(get_db)):
    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    if not day: raise HTTPException(404, "Day not found")
    if day.locked: raise HTTPException(400, "Already locked")
    day.locked = True; day.locked_at = datetime.utcnow()
    db.commit(); return {"status": "day_ended", "locked_at": day.locked_at}

@app.post("/api/day/{log_date}/unlock")
def unlock_day(log_date: date, db: Session = Depends(get_db)):
    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    if not day: raise HTTPException(404, "Day not found")
    day.locked = False; day.locked_at = None
    db.commit(); return {"status": "unlocked"}

@app.get("/api/day/{log_date}/ai-suggestions")
async def get_ai_suggestions(log_date: date, db: Session = Depends(get_db)):
    if not _groq_client:
        return {"habits": ["Morning review (5 min)", "Evening reflection"],
                "tasks": ["Study 1 CO from any subject", "Review yesterday's notes", "Plan tomorrow"]}

    subjects = db.query(Subject).filter(Subject.status == 'active').all()
    recent_days = db.query(DayLog).filter(DayLog.log_date >= log_date - timedelta(days=7)).all()
    avg_tasks = 0.0
    if recent_days:
        totals = [db.query(Entry).filter(Entry.day_log_id == d.id).count() for d in recent_days]
        avg_tasks = sum(totals) / len(totals)

    day = db.query(DayLog).filter(DayLog.log_date == log_date).first()
    existing = [e.text for e in db.query(Entry).filter(Entry.day_log_id == day.id).all()] if day else []

    # Also pull today's timetable slots for smarter suggestions
    dow = log_date.weekday()
    today_slots = db.query(TimetableSlot).filter(TimetableSlot.day_of_week == dow).order_by(TimetableSlot.time_start).all()
    class_info = [f"{s.time_start} {s.subject_name}" for s in today_slots] if today_slots else []

    prompt = (
        f"Generate an effective daily schedule for {log_date.strftime('%A, %B %d')}.\n"
        f"Subjects: {', '.join(s.name for s in subjects[:5]) or 'None set up'}\n"
        f"Today's classes: {', '.join(class_info) or 'No classes scheduled'}\n"
        f"Existing tasks: {existing or 'None yet'}\n"
        f"Avg tasks/day this week: {avg_tasks:.1f}\n\n"
        f"Suggest 2 daily habits (morning routine, review) and 4-6 focused study tasks based on the classes above.\n"
        f"SHORT text (under 10 words each). Be specific — name the subject.\n"
        f'Return JSON only: {{"habits": ["..."], "tasks": ["..."]}}'
    )

    try:
        response = await _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise student productivity assistant. Always return valid JSON only, no markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception:
        return {"habits": ["Morning review (5 min)", "Evening reflection"],
                "tasks": ["Study 1 CO from any subject", "Review yesterday's notes", "Plan tomorrow"]}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — PROGRESS
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/progress/streak")
def get_streak(db: Session = Depends(get_db)):
    today = date.today()
    yesterday = today - timedelta(days=1)
    yday = db.query(DayLog).filter(DayLog.log_date == yesterday).first()
    if yday and not yday.locked:
        return {"streak": 0, "broken_by": "missed_yesterday"}
    streak = 0
    check = today
    tlog = db.query(DayLog).filter(DayLog.log_date == today).first()
    if tlog and tlog.locked: streak = 1; check = yesterday
    else: check = yesterday
    while check >= today - timedelta(days=366):
        d = db.query(DayLog).filter(DayLog.log_date == check).first()
        if not d or not d.locked: break
        streak += 1; check -= timedelta(days=1)
    return {"streak": streak, "as_of": today.isoformat()}

@app.get("/api/progress/performance-chart")
def performance_chart(period: str = "7d", db: Session = Depends(get_db)):
    today = date.today()
    period_map = {"1d": 1, "7d": 7, "15d": 15, "1m": 30, "6m": 180, "1y": 365}
    days_back = period_map.get(period, 7)
    if period == "all":
        first = db.query(DayLog).order_by(DayLog.log_date.asc()).first()
        days_back = (today - first.log_date).days + 1 if first else 30
    start = today - timedelta(days=days_back - 1)
    results = []
    use_weeks = days_back > 60
    if not use_weeks:
        for i in range(days_back):
            d = start + timedelta(days=i)
            day = db.query(DayLog).filter(DayLog.log_date == d).first()
            entries = db.query(Entry).filter(Entry.day_log_id == day.id).all() if day else []
            total = len(entries); done = sum(1 for e in entries if e.done)
            results.append({"label": d.strftime("%a %d"), "date": d.isoformat(),
                            "total_tasks": total, "completed_tasks": done,
                            "completion_percentage": round(done / total * 100, 1) if total else 0})
    else:
        wstart = start; wn = 1
        while wstart <= today:
            wend = min(wstart + timedelta(days=6), today)
            days = db.query(DayLog).filter(DayLog.log_date >= wstart, DayLog.log_date <= wend).all()
            wt = wd = 0
            for day in days:
                ents = db.query(Entry).filter(Entry.day_log_id == day.id).all()
                wt += len(ents); wd += sum(1 for e in ents if e.done)
            results.append({"label": f"W{wn}", "date": wstart.isoformat(),
                            "total_tasks": wt, "completed_tasks": wd,
                            "completion_percentage": round(wd / wt * 100, 1) if wt else 0})
            wstart += timedelta(days=7); wn += 1
    return {"period": period, "start_date": start.isoformat(), "end_date": today.isoformat(), "data": results}

@app.get("/api/progress/analysis")
def performance_analysis(db: Session = Depends(get_db)):
    today = date.today()
    logs = db.query(DayLog).filter(DayLog.log_date >= today - timedelta(days=30)).all()
    if not logs: return {"status": "NO_DATA", "metrics": None, "warnings": ["No data yet. Start logging!"], "recommendations": ["Create today's checklist", "Aim for 5-7 tasks max"]}
    day_map = {}
    for d in logs:
        ents = db.query(Entry).filter(Entry.day_log_id == d.id).all()
        day_map[d.id] = {"date": d.log_date, "locked": d.locked, "total": len(ents), "completed": sum(1 for e in ents if e.done)}
    total_days = len(day_map)
    completed_days = sum(1 for d in day_map.values() if d["locked"])
    total_tasks = sum(d["total"] for d in day_map.values())
    avg_tasks = round(total_tasks / total_days, 1) if total_days else 0
    rates = [d["completed"] / d["total"] for d in day_map.values() if d["total"] > 0]
    avg_comp = round(sum(rates) / len(rates) * 100, 1) if rates else 0
    warnings, recs = [], []
    if avg_tasks > 12 and avg_comp < 60: warnings.append("Overplanning with poor execution"); recs.append("Cut tasks by 30%")
    if avg_comp < 50: warnings.append("Low execution quality"); recs.append("Focus on must-do tasks only")
    if total_days - completed_days >= 6: warnings.append("Too many missed days"); recs.append("Lock days even with 1 task done")
    health = "critical" if avg_comp < 40 else "fragile" if avg_comp < 65 else "stable"
    return {"status": health, "metrics": {"avg_daily_tasks": avg_tasks, "avg_completion_rate": avg_comp,
            "completed_days_30d": completed_days, "skipped_days_30d": total_days - completed_days},
            "warnings": warnings, "recommendations": recs if recs else ["Keep it up! System stable."]}

@app.post("/api/progress/check-and-reset-yesterday")
def check_reset_yesterday(db: Session = Depends(get_db)):
    yesterday = date.today() - timedelta(days=1)
    day = db.query(DayLog).filter(DayLog.log_date == yesterday).first()
    if not day: return {"action": "none", "date": yesterday.isoformat()}
    if day.locked: return {"action": "already_locked", "date": yesterday.isoformat()}
    entries = db.query(Entry).filter(Entry.day_log_id == day.id).all()
    for e in entries: e.done = False
    day.locked = True; day.locked_at = datetime.utcnow()
    db.commit()
    return {"action": "auto_reset", "message": f"Auto-locked yesterday with {len(entries)} tasks", "date": yesterday.isoformat()}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — PROJECTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/projects")
def list_projects(status: str = Query('active'), db: Session = Depends(get_db)):
    q = db.query(Project)
    if status != 'all': q = q.filter(Project.status == status)
    projects = q.order_by(Project.updated_at.desc()).all()
    result = []
    for p in projects:
        tasks = db.query(ProjectTask).filter(ProjectTask.project_id == p.id).all()
        done = sum(1 for t in tasks if t.done)
        result.append({"id": p.id, "name": p.name, "description": p.description,
                       "status": p.status, "priority": p.priority,
                       "deadline": p.deadline.isoformat() if p.deadline else None,
                       "progress": p.progress, "task_count": len(tasks), "done_count": done,
                       "created_at": p.created_at.isoformat(), "tasks": [
                           {"id": t.id, "text": t.text, "done": t.done} for t in tasks]})
    return result

@app.post("/api/projects")
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(**data.dict())
    db.add(p); db.commit(); db.refresh(p)
    return {"id": p.id, "name": p.name}

@app.patch("/api/projects/{pid}")
def update_project(pid: int, data: dict, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == pid).first()
    if not p: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(p, k): setattr(p, k, v)
    p.updated_at = datetime.utcnow()
    db.commit(); return {"ok": True}

@app.delete("/api/projects/{pid}")
def delete_project(pid: int, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == pid).first()
    if not p: raise HTTPException(404, "Not found")
    db.delete(p); db.commit(); return {"ok": True}

@app.post("/api/projects/{pid}/tasks")
def add_project_task(pid: int, data: ProjectTaskCreate, db: Session = Depends(get_db)):
    t = ProjectTask(project_id=pid, text=data.text)
    db.add(t); db.commit(); db.refresh(t)
    return {"id": t.id, "text": t.text, "done": t.done}

@app.patch("/api/projects/tasks/{tid}")
def toggle_project_task(tid: int, data: dict, db: Session = Depends(get_db)):
    t = db.query(ProjectTask).filter(ProjectTask.id == tid).first()
    if not t: raise HTTPException(404, "Not found")
    t.done = data.get("done", t.done)
    tasks = db.query(ProjectTask).filter(ProjectTask.project_id == t.project_id).all()
    done = sum(1 for x in tasks if (x.id == tid and t.done) or (x.id != tid and x.done))
    p = db.query(Project).filter(Project.id == t.project_id).first()
    if p: p.progress = int(done / len(tasks) * 100) if tasks else 0
    db.commit(); return {"ok": True}

@app.delete("/api/projects/tasks/{tid}")
def delete_project_task(tid: int, db: Session = Depends(get_db)):
    t = db.query(ProjectTask).filter(ProjectTask.id == tid).first()
    if not t: raise HTTPException(404, "Not found")
    db.delete(t); db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — NOTES
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/notes")
def list_notes(q: str = Query(None), db: Session = Depends(get_db)):
    query = db.query(Note)
    if q:
        query = query.filter((Note.title.ilike(f'%{q}%')) | (Note.content.ilike(f'%{q}%')) | (Note.tags.ilike(f'%{q}%')))
    return query.order_by(Note.updated_at.desc()).all()

@app.post("/api/notes")
def create_note(data: NoteCreate, db: Session = Depends(get_db)):
    n = Note(**data.dict())
    db.add(n); db.commit(); db.refresh(n)
    return n

@app.patch("/api/notes/{nid}")
def update_note(nid: int, data: dict, db: Session = Depends(get_db)):
    n = db.query(Note).filter(Note.id == nid).first()
    if not n: raise HTTPException(404, "Not found")
    for k, v in data.items():
        if hasattr(n, k): setattr(n, k, v)
    n.updated_at = datetime.utcnow()
    db.commit(); return n

@app.delete("/api/notes/{nid}")
def delete_note(nid: int, db: Session = Depends(get_db)):
    n = db.query(Note).filter(Note.id == nid).first()
    if not n: raise HTTPException(404, "Not found")
    db.delete(n); db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — AI CHAT  (powered by Groq)
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/ai/chat")
async def ai_chat(msg: AIChatMessage, db: Session = Depends(get_db)):
    user_msg = AIChat(role="user", content=msg.message, context=msg.context)
    db.add(user_msg); db.flush()
    response_text = await ask_groq(msg.message, msg.context, db)
    ai_msg = AIChat(role="assistant", content=response_text, context=msg.context)
    db.add(ai_msg); db.commit()
    return {"response": response_text, "message_id": ai_msg.id}

@app.get("/api/ai/chat/history")
def get_chat_history(limit: int = Query(20), db: Session = Depends(get_db)):
    msgs = db.query(AIChat).order_by(AIChat.created_at.desc()).limit(limit).all()
    return list(reversed(msgs))

@app.delete("/api/ai/chat/history")
def clear_chat_history(db: Session = Depends(get_db)):
    db.query(AIChat).delete(); db.commit(); return {"ok": True}

# ══════════════════════════════════════════════════════════════════════
# API ROUTES — DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    today = date.today()
    day = db.query(DayLog).filter(DayLog.log_date == today).first()
    habits = tasks = done_habits = done_tasks = 0
    if day:
        hs = db.query(Entry).filter(Entry.day_log_id == day.id, Entry.entry_type == "habit").all()
        ts = db.query(Entry).filter(Entry.day_log_id == day.id, Entry.entry_type == "task").all()
        habits = len(hs); tasks = len(ts)
        done_habits = sum(1 for e in hs if e.done); done_tasks = sum(1 for e in ts if e.done)
    week_start = today - timedelta(days=7)
    week_entries = db.query(Entry).join(DayLog).filter(DayLog.log_date >= week_start).all()
    week_done = sum(1 for e in week_entries if e.done)
    active_sems = db.query(Semester).filter(Semester.status == 'active').count()
    active_subs = db.query(Subject).filter(Subject.status == 'active').count()
    active_projs = db.query(Project).filter(Project.status == 'active').count()
    notes_count = db.query(Note).count()
    streak_data = get_streak(db)
    month_days = db.query(DayLog).filter(DayLog.log_date >= today.replace(day=1), DayLog.locked == True).count()
    month_total = today.day
    return {
        "today": {"date": today.isoformat(), "day_locked": day.locked if day else False,
                  "habits": habits, "tasks": tasks, "done_habits": done_habits, "done_tasks": done_tasks},
        "streak": streak_data["streak"],
        "week_tasks_done": week_done, "week_tasks_total": len(week_entries),
        "active_semesters": active_sems, "active_subjects": active_subs,
        "active_projects": active_projs, "notes_count": notes_count,
        "month_completion": round(month_days / month_total * 100) if month_total else 0,
        "groq_enabled": bool(_groq_client),
        "groq_model": GROQ_MODEL,
        "timetable_today": [
            {"time_start": s.time_start, "time_end": s.time_end,
             "subject_name": s.subject_name, "room": s.room or ""}
            for s in db.query(TimetableSlot).filter(
                TimetableSlot.day_of_week == date.today().weekday()
            ).order_by(TimetableSlot.time_start).all()
        ]
    }

# ══════════════════════════════════════════════════════════════════════
# FRONTEND — SINGLE PAGE APP (FULL INLINE HTML)
# ══════════════════════════════════════════════════════════════════════

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DayMark — Student OS</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0a0a0f;
  --bg2: #111118;
  --bg3: #18181f;
  --bg4: #1e1e28;
  --border: rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.12);
  --text: #f0f0f5;
  --text2: #9090a8;
  --text3: #5a5a72;
  --accent: #7c6ef5;
  --accent2: #5b4fd6;
  --green: #22d3a0;
  --green2: #16a67a;
  --yellow: #f5c842;
  --red: #f55e5e;
  --orange: #f5934a;
  --blue: #4aadf5;
  --pink: #f54ab0;
  --r: 12px;
  --r2: 18px;
  --sidebar: 240px;
  --shadow: 0 8px 32px rgba(0,0,0,0.5);
  --transition: 0.2s ease;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; }
body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; font-size: 15px; line-height: 1.6; overflow-x: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }

/* SIDEBAR */
.sidebar { width: var(--sidebar); min-height: 100vh; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; position: fixed; left: 0; top: 0; z-index: 100; transition: transform var(--transition); }
.sidebar-logo { padding: 24px 20px 16px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--border); }
.sidebar-logo .logo-mark { width: 32px; height: 32px; background: linear-gradient(135deg, var(--accent), var(--green)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.sidebar-logo span { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 18px; letter-spacing: -0.5px; }
.sidebar-logo .version { font-size: 10px; color: var(--text3); margin-left: 2px; }
.nav-section { padding: 16px 12px 8px; }
.nav-label { font-size: 10px; font-weight: 600; color: var(--text3); letter-spacing: 1.5px; text-transform: uppercase; padding: 0 8px; margin-bottom: 6px; }
.nav-item { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: var(--r); cursor: pointer; color: var(--text2); font-size: 14px; font-weight: 400; transition: all var(--transition); user-select: none; border: none; background: none; width: 100%; text-align: left; }
.nav-item:hover { background: var(--bg3); color: var(--text); }
.nav-item.active { background: linear-gradient(135deg, rgba(124,110,245,0.15), rgba(34,211,160,0.08)); color: var(--text); font-weight: 500; }
.nav-item .nav-icon { width: 20px; text-align: center; font-size: 15px; }
.nav-item .nav-badge { margin-left: auto; background: var(--accent); color: white; border-radius: 20px; padding: 1px 7px; font-size: 11px; font-weight: 600; }
.sidebar-streak { padding: 16px 12px; margin-top: auto; }
.streak-card { background: linear-gradient(135deg, rgba(245,200,66,0.1), rgba(245,147,74,0.08)); border: 1px solid rgba(245,200,66,0.2); border-radius: var(--r); padding: 12px 14px; display: flex; align-items: center; gap: 12px; }
.streak-num { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: var(--yellow); line-height: 1; }
.streak-label { font-size: 12px; color: var(--text2); }
.sidebar-bottom { padding: 12px; border-top: 1px solid var(--border); }
.groq-badge { display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: rgba(124,110,245,0.1); border-radius: var(--r); font-size: 12px; color: var(--text2); }
.groq-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }

/* MAIN */
.main { margin-left: var(--sidebar); flex: 1; min-height: 100vh; display: flex; flex-direction: column; }
.page { display: none; flex-direction: column; flex: 1; padding: 0; }
.page.active { display: flex; }
.page-header { padding: 28px 32px 0; display: flex; align-items: center; justify-content: space-between; }
.page-title { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
.page-subtitle { color: var(--text2); font-size: 14px; margin-top: 2px; }
.page-body { padding: 24px 32px; flex: 1; overflow-y: auto; }

/* CARDS */
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r2); padding: 20px; }
.card-sm { padding: 14px 16px; }
.card-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 15px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }

/* GRID */
.grid { display: grid; gap: 16px; }
.grid-2 { grid-template-columns: 1fr 1fr; }
.grid-3 { grid-template-columns: 1fr 1fr 1fr; }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

/* STAT CARDS */
.stat-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--r2); padding: 18px 20px; transition: all var(--transition); }
.stat-card:hover { border-color: var(--border2); transform: translateY(-2px); }
.stat-value { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; line-height: 1.1; }
.stat-label { font-size: 13px; color: var(--text2); margin-top: 4px; }
.stat-icon { font-size: 20px; margin-bottom: 8px; }

/* BUTTONS */
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 9px 18px; border-radius: var(--r); border: none; cursor: pointer; font-size: 14px; font-weight: 500; font-family: 'DM Sans', sans-serif; transition: all var(--transition); white-space: nowrap; }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: var(--accent2); transform: translateY(-1px); box-shadow: 0 4px 20px rgba(124,110,245,0.4); }
.btn-green { background: var(--green); color: var(--bg); }
.btn-green:hover { background: var(--green2); }
.btn-ghost { background: var(--bg3); color: var(--text2); border: 1px solid var(--border); }
.btn-ghost:hover { background: var(--bg4); color: var(--text); }
.btn-danger { background: rgba(245,94,94,0.15); color: var(--red); border: 1px solid rgba(245,94,94,0.2); }
.btn-danger:hover { background: rgba(245,94,94,0.25); }
.btn-sm { padding: 6px 12px; font-size: 13px; }
.btn-xs { padding: 4px 8px; font-size: 12px; }
.btn-icon { width: 34px; height: 34px; padding: 0; border-radius: 8px; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

/* INPUTS */
.input, .textarea, .select { width: 100%; padding: 10px 14px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--r); color: var(--text); font-size: 14px; font-family: 'DM Sans', sans-serif; outline: none; transition: border-color var(--transition); }
.input:focus, .textarea:focus, .select:focus { border-color: var(--accent); }
.textarea { resize: vertical; min-height: 80px; }
.select option { background: var(--bg3); }
.input-group { display: flex; gap: 8px; }
.form-row { display: grid; gap: 14px; margin-bottom: 14px; }
.form-row.cols-2 { grid-template-columns: 1fr 1fr; }
.form-row.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
label { font-size: 13px; color: var(--text2); margin-bottom: 6px; display: block; }

/* BADGE */
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.badge-green { background: rgba(34,211,160,0.15); color: var(--green); border: 1px solid rgba(34,211,160,0.2); }
.badge-yellow { background: rgba(245,200,66,0.15); color: var(--yellow); border: 1px solid rgba(245,200,66,0.2); }
.badge-red { background: rgba(245,94,94,0.15); color: var(--red); border: 1px solid rgba(245,94,94,0.2); }
.badge-purple { background: rgba(124,110,245,0.15); color: var(--accent); border: 1px solid rgba(124,110,245,0.2); }
.badge-blue { background: rgba(74,173,245,0.15); color: var(--blue); border: 1px solid rgba(74,173,245,0.2); }
.badge-orange { background: rgba(245,147,74,0.15); color: var(--orange); border: 1px solid rgba(245,147,74,0.2); }

/* TASK ITEMS */
.task-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 6px; group: true; transition: all var(--transition); }
.task-item:hover { border-color: var(--border2); }
.task-item.done { opacity: 0.55; }
.task-item.done .task-text { text-decoration: line-through; color: var(--text2); }
.task-cb { width: 18px; height: 18px; border: 2px solid var(--border2); border-radius: 5px; cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; transition: all var(--transition); }
.task-cb.checked { background: var(--green); border-color: var(--green); }
.task-cb.checked::after { content: '✓'; color: var(--bg); font-size: 11px; font-weight: 700; }
.habit-cb.checked { background: var(--accent); border-color: var(--accent); }
.task-text { flex: 1; font-size: 14px; }
.task-meta { font-size: 11px; color: var(--text3); margin-top: 2px; }
.task-actions { display: flex; gap: 4px; opacity: 0; transition: opacity var(--transition); }
.task-item:hover .task-actions { opacity: 1; }

/* PROGRESS BAR */
.progress-bar { height: 6px; background: var(--bg4); border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.progress-green { background: var(--green); }
.progress-yellow { background: var(--yellow); }
.progress-red { background: var(--red); }
.progress-purple { background: var(--accent); }

/* MODAL */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); z-index: 1000; display: none; align-items: center; justify-content: center; padding: 20px; }
.modal-overlay.open { display: flex; }
.modal { background: var(--bg2); border: 1px solid var(--border2); border-radius: 20px; padding: 28px; max-width: 520px; width: 100%; max-height: 90vh; overflow-y: auto; animation: modalIn 0.2s ease; }
.modal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.modal-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; }
.modal-close { background: var(--bg3); border: none; color: var(--text2); cursor: pointer; width: 30px; height: 30px; border-radius: 8px; font-size: 18px; display: flex; align-items: center; justify-content: center; }
.modal-close:hover { background: var(--bg4); color: var(--text); }
.modal-footer { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; border-top: 1px solid var(--border); padding-top: 16px; }

/* TOAST */
.toast-container { position: fixed; top: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 8px; }
.toast { background: var(--bg2); border: 1px solid var(--border2); border-radius: var(--r); padding: 12px 16px; display: flex; align-items: center; gap: 10px; font-size: 14px; min-width: 260px; max-width: 360px; box-shadow: var(--shadow); animation: toastIn 0.3s ease; }
.toast.success { border-left: 3px solid var(--green); }
.toast.error { border-left: 3px solid var(--red); }
.toast.info { border-left: 3px solid var(--accent); }
.toast.warning { border-left: 3px solid var(--yellow); }
.toast-icon { font-size: 16px; }
.toast-msg { flex: 1; }
.toast-close { background: none; border: none; color: var(--text3); cursor: pointer; font-size: 16px; }

/* MISC */
.divider { height: 1px; background: var(--border); margin: 16px 0; }
.empty-state { text-align: center; padding: 48px 24px; color: var(--text2); }
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.empty-desc { color: var(--text2); font-size: 14px; }
.loading { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 24px; color: var(--text2); }
.spinner { width: 20px; height: 20px; border: 2px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
.tab-bar { display: flex; gap: 4px; background: var(--bg3); border-radius: var(--r); padding: 4px; margin-bottom: 20px; }
.tab { flex: 1; padding: 8px 14px; border-radius: 8px; border: none; background: none; color: var(--text2); cursor: pointer; font-size: 13px; font-weight: 500; font-family: 'DM Sans', sans-serif; transition: all var(--transition); }
.tab.active { background: var(--bg2); color: var(--text); box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.tag { display: inline-block; padding: 2px 8px; background: var(--bg4); border-radius: 6px; font-size: 12px; color: var(--text2); margin: 2px; cursor: pointer; }
.tag:hover { background: rgba(124,110,245,0.2); color: var(--accent); }
.accordion { border: 1px solid var(--border); border-radius: var(--r2); overflow: hidden; margin-bottom: 10px; }
.accordion-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; cursor: pointer; user-select: none; background: var(--bg2); transition: background var(--transition); }
.accordion-header:hover { background: var(--bg3); }
.accordion-title { font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; }
.accordion-body { padding: 0 18px; max-height: 0; overflow: hidden; transition: max-height 0.3s ease, padding 0.3s ease; }
.accordion.open .accordion-body { max-height: 2000px; padding: 16px 18px; }
.accordion-arrow { color: var(--text3); transition: transform 0.3s; }
.accordion.open .accordion-arrow { transform: rotate(180deg); }
.chip-group { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { padding: 5px 12px; border-radius: 20px; border: 1px solid var(--border); background: var(--bg3); color: var(--text2); cursor: pointer; font-size: 13px; transition: all var(--transition); }
.chip.selected { background: rgba(124,110,245,0.2); border-color: var(--accent); color: var(--accent); }

/* CHAT */
.chat-container { display: flex; flex-direction: column; height: calc(100vh - 160px); }
.chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.chat-msg { max-width: 80%; animation: fadeIn 0.3s ease; }
.chat-msg.user { align-self: flex-end; }
.chat-msg.assistant { align-self: flex-start; }
.chat-bubble { padding: 12px 16px; border-radius: 16px; font-size: 14px; line-height: 1.6; }
.chat-msg.user .chat-bubble { background: var(--accent); color: white; border-bottom-right-radius: 4px; }
.chat-msg.assistant .chat-bubble { background: var(--bg3); border: 1px solid var(--border); border-bottom-left-radius: 4px; white-space: pre-wrap; }
.chat-input-area { padding: 16px; border-top: 1px solid var(--border); display: flex; gap: 10px; background: var(--bg2); }
.chat-input { flex: 1; padding: 12px 16px; background: var(--bg3); border: 1px solid var(--border); border-radius: 12px; color: var(--text); font-size: 14px; font-family: 'DM Sans', sans-serif; resize: none; outline: none; max-height: 120px; }
.chat-input:focus { border-color: var(--accent); }
.chat-send { width: 44px; height: 44px; background: var(--accent); border: none; border-radius: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; transition: all var(--transition); flex-shrink: 0; }
.chat-send:hover { background: var(--accent2); transform: scale(1.05); }
.chat-typing { display: flex; align-items: center; gap: 6px; padding: 12px 16px; background: var(--bg3); border-radius: 16px; max-width: 80px; }
.typing-dot { width: 7px; height: 7px; background: var(--text3); border-radius: 50%; animation: typingBounce 1.2s infinite; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* CALENDAR */
.calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
.cal-day { aspect-ratio: 1; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 12px; font-weight: 500; cursor: pointer; transition: all var(--transition); position: relative; }
.cal-day .day-num { font-size: 13px; }
.cal-day.today { border: 2px solid var(--accent); }
.cal-day.locked { background: rgba(34,211,160,0.15); color: var(--green); }
.cal-day.missed { background: rgba(245,94,94,0.1); color: var(--red); }
.cal-day.future { color: var(--text3); cursor: default; }
.cal-day.empty { pointer-events: none; }
.cal-header { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; margin-bottom: 6px; }
.cal-header span { text-align: center; font-size: 11px; color: var(--text3); font-weight: 600; padding: 4px; }

/* PRIORITY */
.priority-high { color: var(--red); }
.priority-medium { color: var(--yellow); }
.priority-low { color: var(--green); }

/* AI SUGGESTIONS */
.suggestion-pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; background: rgba(124,110,245,0.1); border: 1px solid rgba(124,110,245,0.25); border-radius: 20px; font-size: 13px; cursor: pointer; color: var(--text); margin: 3px; transition: all var(--transition); }
.suggestion-pill:hover { background: rgba(124,110,245,0.2); transform: translateY(-1px); }
.suggestion-pill .pill-icon { font-size: 14px; }
.suggestion-pill .pill-add { font-size: 16px; color: var(--accent); margin-left: 2px; }

/* ANIMATIONS */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes modalIn { from { opacity: 0; transform: scale(0.95) translateY(10px); } to { opacity: 1; transform: scale(1) translateY(0); } }
@keyframes toastIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
@keyframes typingBounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.slide-up { animation: slideUp 0.4s ease both; }

/* MOBILE */
.mobile-bar { display: none; position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg2); border-top: 1px solid var(--border); z-index: 200; padding: 8px; }
.mobile-bar-items { display: flex; justify-content: space-around; }
.mobile-bar-item { display: flex; flex-direction: column; align-items: center; gap: 3px; padding: 6px 12px; border-radius: var(--r); cursor: pointer; color: var(--text2); font-size: 10px; border: none; background: none; transition: all var(--transition); }
.mobile-bar-item.active { color: var(--accent); }
.mobile-bar-item .m-icon { font-size: 20px; }
@media (max-width: 768px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); }
  .main { margin-left: 0; }
  .mobile-bar { display: flex; }
  .page-body { padding: 16px; padding-bottom: 80px; }
  .grid-4 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr; }
  .grid-2 { grid-template-columns: 1fr; }
  .page-header { padding: 20px 16px 0; }
}

/* HEATMAP */
.heatmap { display: flex; flex-direction: column; gap: 3px; }
.heatmap-row { display: flex; gap: 3px; }
.heatmap-cell { width: 14px; height: 14px; border-radius: 3px; background: var(--bg4); transition: all var(--transition); cursor: default; }
.heatmap-cell.l1 { background: rgba(34,211,160,0.2); }
.heatmap-cell.l2 { background: rgba(34,211,160,0.4); }
.heatmap-cell.l3 { background: rgba(34,211,160,0.65); }
.heatmap-cell.l4 { background: rgba(34,211,160,0.9); }
.heatmap-cell.today-cell { outline: 2px solid var(--accent); }

/* SEMESTER TREE */
.sem-tree { margin-bottom: 10px; }
.sem-tree-subject { padding: 10px 14px; background: var(--bg3); border-radius: 10px; margin-bottom: 6px; }
.sem-tree-co { padding: 6px 12px 6px 28px; background: var(--bg4); border-radius: 8px; margin: 4px 0; font-size: 13px; color: var(--text2); display: flex; align-items: center; gap: 8px; }

/* SECTION */
.section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 17px; font-weight: 700; }

/* STATUS INDICATOR */
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-active { background: var(--green); }
.status-archived { background: var(--text3); }
.status-hidden { background: var(--yellow); }

/* DATE NAV */
.date-nav { display: flex; align-items: center; gap: 12px; }
.date-nav-btn { width: 32px; height: 32px; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer; color: var(--text2); font-size: 16px; transition: all var(--transition); }
.date-nav-btn:hover { background: var(--bg4); color: var(--text); }
.date-display { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; min-width: 180px; text-align: center; }

/* LOCKED OVERLAY */
.locked-banner { display: flex; align-items: center; gap: 10px; padding: 12px 16px; background: rgba(34,211,160,0.08); border: 1px solid rgba(34,211,160,0.2); border-radius: var(--r); margin-bottom: 16px; font-size: 14px; color: var(--green); }

/* AI CHAT QUICK ACTIONS */
.ai-quick { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; padding: 0 16px; }
.ai-quick-btn { padding: 6px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: 20px; font-size: 12px; color: var(--text2); cursor: pointer; transition: all var(--transition); }
.ai-quick-btn:hover { background: var(--bg4); color: var(--text); border-color: var(--accent); }

/* RESOURCE UPLOAD */
.upload-zone { border: 2px dashed var(--border2); border-radius: var(--r2); padding: 32px; text-align: center; cursor: pointer; transition: all var(--transition); }
.upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: rgba(124,110,245,0.05); }
.upload-zone .upload-icon { font-size: 40px; margin-bottom: 10px; }
.upload-zone .upload-text { color: var(--text2); font-size: 14px; }

/* CHART CONTAINER */
.chart-wrap { position: relative; height: 220px; }
</style>
</head>
<body>

<!-- TOAST CONTAINER -->
<div class="toast-container" id="toastContainer"></div>

<!-- SIDEBAR -->
<nav class="sidebar" id="sidebar">
  <div class="sidebar-logo">
    <div class="logo-mark">📚</div>
    <span>DayMark <span class="version">v3</span></span>
  </div>

  <div class="nav-section">
    <div class="nav-label">Main</div>
    <button class="nav-item active" data-page="dashboard">
      <span class="nav-icon">🏠</span> Dashboard
    </button>
    <button class="nav-item" data-page="today">
      <span class="nav-icon">✅</span> Today
    </button>
    <button class="nav-item" data-page="progress">
      <span class="nav-icon">📊</span> Progress
    </button>
  </div>

  <div class="nav-section">
    <div class="nav-label">Academic</div>
    <button class="nav-item" data-page="semesters">
      <span class="nav-icon">🎓</span> Semesters
    </button>
    <button class="nav-item" data-page="subjects">
      <span class="nav-icon">📖</span> Subjects & COs
    </button>
    <button class="nav-item" data-page="timetable">
      <span class="nav-icon">🗓️</span> Timetable
    </button>
    <button class="nav-item" data-page="resources">
      <span class="nav-icon">📄</span> Resources
    </button>
  </div>

  <div class="nav-section">
    <div class="nav-label">Life</div>
    <button class="nav-item" data-page="projects">
      <span class="nav-icon">🚀</span> Projects
    </button>
    <button class="nav-item" data-page="notes">
      <span class="nav-icon">📝</span> Notes
    </button>
    <button class="nav-item" data-page="ai">
      <span class="nav-icon">🤖</span> AI Assistant
    </button>
  </div>

  <div class="sidebar-streak">
    <div class="streak-card">
      <span style="font-size:28px">🔥</span>
      <div>
        <div class="streak-num" id="sidebarStreak">0</div>
        <div class="streak-label">Day Streak</div>
      </div>
    </div>
  </div>

  <div class="sidebar-bottom">
    <div class="groq-badge">
      <div class="groq-dot" id="groqDot"></div>
      <span id="groqStatus">Checking AI...</span>
    </div>
  </div>
</nav>

<!-- MAIN CONTENT -->
<main class="main">

  <!-- ═══════════════════════════════════════ DASHBOARD ═══════════════════════════════════════ -->
  <div class="page active" id="page-dashboard">
    <div class="page-header">
      <div>
        <div class="page-title" id="dashGreeting">Good Morning! 👋</div>
        <div class="page-subtitle" id="dashDate">Loading...</div>
      </div>
      <button class="btn btn-primary" onclick="navigate('today')">Plan Today →</button>
    </div>
    <div class="page-body">
      <!-- Stats -->
      <div class="grid grid-4 slide-up" style="margin-bottom:20px" id="dashStats">
        <div class="stat-card"><div class="stat-icon">🔥</div><div class="stat-value" id="dStreak">0</div><div class="stat-label">Day Streak</div></div>
        <div class="stat-card"><div class="stat-icon">✅</div><div class="stat-value" id="dMonth">0%</div><div class="stat-label">Month Done</div></div>
        <div class="stat-card"><div class="stat-icon">📚</div><div class="stat-value" id="dSubs">0</div><div class="stat-label">Active Subjects</div></div>
        <div class="stat-card"><div class="stat-icon">⚡</div><div class="stat-value" id="dWeek">0</div><div class="stat-label">Tasks This Week</div></div>
      </div>

      <div class="grid grid-2" style="margin-bottom:20px">
        <!-- Today snapshot -->
        <div class="card">
          <div class="card-title">📅 Today's Snapshot</div>
          <div id="dashTodaySnap">
            <div class="loading"><div class="spinner"></div> Loading...</div>
          </div>
        </div>
      </div>
      <!-- Today timetable + quick actions row -->
      <div class="grid grid-2" style="margin-bottom:20px">
        <div class="card">
          <div class="section-header">
            <div class="card-title" style="margin:0">🏫 Today's Class Schedule</div>
            <button class="btn btn-ghost btn-sm" onclick="navigate('timetable')">Edit</button>
          </div>
          <div id="dashTimetable"><div style="color:var(--text3);font-size:13px;padding:8px 0">No classes scheduled. <span style="cursor:pointer;color:var(--accent)" onclick="navigate('timetable')">Set up timetable →</span></div></div>
        </div>
        <!-- Quick actions -->
        <div class="card">
          <div class="card-title">⚡ Quick Actions</div>
          <div style="display:grid;gap:8px">
            <button class="btn btn-ghost" style="justify-content:flex-start" onclick="generateFromTimetable();navigate('today')">🗓️ Auto-Fill Today from Timetable</button>
            <button class="btn btn-ghost" style="justify-content:flex-start" onclick="navigate('today')">✅ Plan / Check Today</button>
            <button class="btn btn-ghost" style="justify-content:flex-start" onclick="openSubjectImportModal()">📤 Import Subject from PDF</button>
            <button class="btn btn-ghost" style="justify-content:flex-start" onclick="navigate('timetable')">🗓️ Manage Timetable</button>
            <button class="btn btn-ghost" style="justify-content:flex-start" onclick="navigate('ai')">🤖 Ask AI Assistant</button>
          </div>
        </div>
      </div>

      <!-- Heatmap -->
      <div class="card" style="margin-bottom:20px">
        <div class="card-title">📈 Activity Heatmap (Last 12 Weeks)</div>
        <div id="heatmapContainer" style="overflow-x:auto"></div>
        <div style="display:flex;gap:6px;align-items:center;margin-top:10px;font-size:12px;color:var(--text2)">
          Less <div style="width:12px;height:12px;border-radius:3px;background:var(--bg4)"></div>
          <div style="width:12px;height:12px;border-radius:3px;background:rgba(34,211,160,0.25)"></div>
          <div style="width:12px;height:12px;border-radius:3px;background:rgba(34,211,160,0.5)"></div>
          <div style="width:12px;height:12px;border-radius:3px;background:rgba(34,211,160,0.9)"></div>
          More
        </div>
      </div>

      <!-- Active projects preview -->
      <div class="card">
        <div class="section-header">
          <div class="card-title" style="margin:0">🚀 Active Projects</div>
          <button class="btn btn-ghost btn-sm" onclick="navigate('projects')">View All</button>
        </div>
        <div id="dashProjects"><div class="loading"><div class="spinner"></div></div></div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ TODAY ═══════════════════════════════════════ -->
  <div class="page" id="page-today">
    <div class="page-header">
      <div>
        <div class="page-title">Daily Execution</div>
        <div class="date-nav" style="margin-top:8px">
          <div class="date-nav-btn" onclick="changeDay(-1)">‹</div>
          <div class="date-display" id="todayDateDisplay">Today</div>
          <div class="date-nav-btn" onclick="changeDay(1)">›</div>
          <button class="btn btn-ghost btn-sm" onclick="goToday()">Today</button>
        </div>
      </div>
      <div style="display:flex;gap:8px" id="todayActions">
        <button class="btn btn-ghost btn-sm" onclick="generateFromTimetable()" id="genTimetableBtn">🗓️ Auto-Fill from Timetable</button>
        <button class="btn btn-ghost btn-sm" onclick="loadAISuggestions()" id="aiSuggestBtn">🤖 AI Suggest</button>
        <button class="btn btn-green" onclick="endDay()" id="endDayBtn">🔒 Lock Day</button>
      </div>
    </div>
    <div class="page-body">
      <div id="lockedBanner" style="display:none">
        <div class="locked-banner">🔒 This day is locked. All tasks saved.
          <button class="btn btn-ghost btn-xs" onclick="unlockDay()" style="margin-left:auto">Unlock</button>
        </div>
      </div>

      <!-- AI Suggestions -->
      <div id="aiSuggestionsBox" style="display:none;margin-bottom:16px">
        <div class="card">
          <div class="card-title">🤖 AI Suggestions <span style="font-size:12px;font-weight:400;color:var(--text2)">Click + to add</span>
            <button class="btn btn-ghost btn-xs" onclick="document.getElementById('aiSuggestionsBox').style.display='none'" style="margin-left:auto">✕</button>
          </div>
          <div id="aiSuggestContent"><div class="loading"><div class="spinner"></div> Thinking...</div></div>
        </div>
      </div>

      <!-- Today's Class Schedule -->
      <div class="card" style="margin-bottom:16px" id="todayClassesCard">
        <div class="section-header">
          <div class="card-title" style="margin:0">🏫 Today's Classes</div>
          <button class="btn btn-ghost btn-sm" onclick="navigate('timetable')">Edit Timetable</button>
        </div>
        <div id="todayClassesList"><div style="color:var(--text3);font-size:13px">No classes set for today. <span style="cursor:pointer;color:var(--accent)" onclick="navigate('timetable')">Set up your timetable →</span></div></div>
      </div>

      <div class="grid grid-2">
        <!-- HABITS -->
        <div>
          <div class="section-header">
            <div class="section-title">🌅 Habits</div>
            <button class="btn btn-ghost btn-sm" onclick="addEntry('habit')" id="addHabitBtn">+ Add</button>
          </div>
          <div id="habitsList"></div>
        </div>
        <!-- TASKS -->
        <div>
          <div class="section-header">
            <div class="section-title">📋 Tasks</div>
            <button class="btn btn-ghost btn-sm" onclick="addEntry('task')" id="addTaskBtn">+ Add</button>
          </div>
          <div id="tasksList"></div>
        </div>
      </div>

      <!-- Progress -->
      <div class="card" style="margin-top:16px" id="todayProgressCard">
        <div class="card-title">Today's Progress</div>
        <div id="todayProgressBar"></div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ PROGRESS ═══════════════════════════════════════ -->
  <div class="page" id="page-progress">
    <div class="page-header">
      <div>
        <div class="page-title">Progress Analytics</div>
        <div class="page-subtitle">Track your consistency over time</div>
      </div>
    </div>
    <div class="page-body">
      <div class="grid grid-4" style="margin-bottom:20px" id="progressStats"></div>

      <!-- Period tabs + chart -->
      <div class="card" style="margin-bottom:20px">
        <div class="section-header">
          <div class="card-title" style="margin:0">📊 Completion Chart</div>
          <div class="tab-bar" style="margin:0;width:auto">
            <button class="tab active" onclick="loadChart('7d',this)">7D</button>
            <button class="tab" onclick="loadChart('15d',this)">15D</button>
            <button class="tab" onclick="loadChart('1m',this)">1M</button>
            <button class="tab" onclick="loadChart('6m',this)">6M</button>
            <button class="tab" onclick="loadChart('1y',this)">1Y</button>
          </div>
        </div>
        <div class="chart-wrap"><canvas id="progressChart"></canvas></div>
      </div>

      <!-- Analysis -->
      <div class="grid grid-2">
        <div class="card" id="analysisCard">
          <div class="card-title">🔍 Performance Analysis</div>
          <div id="analysisContent"><div class="loading"><div class="spinner"></div></div></div>
        </div>
        <div class="card">
          <div class="card-title">📅 Calendar View</div>
          <div id="calendarView"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ SEMESTERS ═══════════════════════════════════════ -->
  <div class="page" id="page-semesters">
    <div class="page-header">
      <div>
        <div class="page-title">Semesters</div>
        <div class="page-subtitle">Manage your academic timeline</div>
      </div>
      <button class="btn btn-primary" onclick="openModal('semModal')">+ New Semester</button>
    </div>
    <div class="page-body">
      <div class="tab-bar" style="width:fit-content">
        <button class="tab active" onclick="loadSemesters('active',this)">Active</button>
        <button class="tab" onclick="loadSemesters('hidden',this)">Hidden</button>
        <button class="tab" onclick="loadSemesters('archived',this)">Archived</button>
        <button class="tab" onclick="loadSemesters('all',this)">All</button>
      </div>
      <div id="semestersList"><div class="loading"><div class="spinner"></div></div></div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ SUBJECTS ═══════════════════════════════════════ -->
  <div class="page" id="page-subjects">
    <div class="page-header">
      <div>
        <div class="page-title">Subjects & Course Outcomes</div>
        <div class="page-subtitle">Manage your courses and learning goals</div>
      </div>
      <div style="display:flex;gap:8px">
        <button class="btn btn-ghost" onclick="openSubjectImportModal()">📤 Import from PDF</button>
        <button class="btn btn-primary" onclick="openModal('subjectModal')">+ Add Manually</button>
      </div>
    </div>
    <div class="page-body">
      <!-- PDF Auto-Import Banner -->
      <div class="card" style="margin-bottom:16px;border:1px dashed var(--accent);background:rgba(124,110,245,0.04)">
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <div style="font-size:28px">🤖</div>
          <div style="flex:1">
            <div style="font-weight:600;margin-bottom:2px">Auto-Import Subjects from Syllabus PDF</div>
            <div style="font-size:13px;color:var(--text2)">Upload any syllabus PDF → Groq AI reads it and automatically creates the subject + all course outcomes instantly.</div>
          </div>
          <button class="btn btn-primary" onclick="openSubjectImportModal()">📤 Import PDF</button>
        </div>
      </div>
      <div id="subjectFilter" style="margin-bottom:16px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="font-size:14px;color:var(--text2);line-height:36px">Semester:</span>
        <div id="semFilterChips" class="chip-group"></div>
      </div>
      <div id="subjectsList"><div class="loading"><div class="spinner"></div></div></div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ RESOURCES ═══════════════════════════════════════ -->
  <div class="page" id="page-resources">
    <div class="page-header">
      <div>
        <div class="page-title">Resources & PDFs</div>
        <div class="page-subtitle">Upload syllabi and extract course outcomes with Groq AI ⚡</div>
      </div>
    </div>
    <div class="page-body">
      <div class="card" style="margin-bottom:20px">
        <div class="card-title">📤 Upload Syllabus PDF</div>
        <div class="form-row cols-2">
          <div>
            <label>Select Subject</label>
            <select class="select" id="uploadSubjectSelect">
              <option value="">-- Select a subject first --</option>
            </select>
          </div>
          <div>
            <label>PDF File</label>
            <input type="file" accept=".pdf" id="pdfFileInput" style="display:none" onchange="handleFileUpload(this)">
            <button class="btn btn-ghost" onclick="document.getElementById('pdfFileInput').click()" style="width:100%">📎 Choose PDF</button>
          </div>
        </div>
        <div id="uploadZone" class="upload-zone" ondragover="e.preventDefault();this.classList.add('dragover')" ondragleave="this.classList.remove('dragover')" ondrop="handleDrop(event)">
          <div class="upload-icon">📄</div>
          <div class="upload-text">Drop PDF here or click button above</div>
        </div>
        <div id="uploadProgress" style="display:none;margin-top:12px">
          <div class="loading"><div class="spinner"></div> Uploading and extracting with Groq AI ⚡...</div>
        </div>
      </div>
      <div id="extractedCOs" style="display:none"></div>
      <div id="resourcesList"></div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ TIMETABLE ════════════════════════════════════════ -->
  <div class="page" id="page-timetable">
    <div class="page-header">
      <div>
        <div class="page-title">🗓️ Class Timetable</div>
        <div class="page-subtitle">Upload your routine PDF — AI builds your weekly schedule automatically</div>
      </div>
      <div style="display:flex;gap:8px">
        <button class="btn btn-ghost btn-sm" onclick="clearTimetable()" style="color:var(--red)">🗑️ Clear All</button>
        <button class="btn btn-ghost" onclick="document.getElementById('routinePdfInput').click()">📤 Import Routine PDF</button>
        <button class="btn btn-primary" onclick="openAddSlotModal()">+ Add Slot</button>
        <input type="file" id="routinePdfInput" accept=".pdf" style="display:none" onchange="importRoutinePdf(this)">
      </div>
    </div>
    <div class="page-body">
      <!-- Import zone -->
      <div class="card" style="margin-bottom:20px;border:1px dashed var(--accent);background:rgba(124,110,245,0.04)" id="timetableImportZone">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
          <div style="font-size:40px">🤖</div>
          <div style="flex:1">
            <div style="font-weight:700;font-size:16px;margin-bottom:4px">Auto-Import Your Class Routine</div>
            <div style="color:var(--text2);font-size:13px;line-height:1.5">Upload your college routine PDF → Groq AI reads it and builds your complete weekly timetable with subject names, times, and room numbers — all automatically.</div>
          </div>
          <button class="btn btn-primary btn-lg" onclick="document.getElementById('routinePdfInput').click()">📤 Upload Routine PDF</button>
        </div>
        <div id="timetableImportProgress" style="display:none;margin-top:12px">
          <div class="loading"><div class="spinner"></div> <span id="timetableImportMsg">Reading PDF with Groq AI...</span></div>
        </div>
      </div>
      <!-- Weekly grid -->
      <div class="card">
        <div class="card-title">📅 Weekly Schedule</div>
        <div id="weeklyTimetable"><div class="loading"><div class="spinner"></div></div></div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ PROJECTS ═══════════════════════════════════════ -->
  <div class="page" id="page-projects">
    <div class="page-header">
      <div>
        <div class="page-title">Projects</div>
        <div class="page-subtitle">Track everything you're building</div>
      </div>
      <button class="btn btn-primary" onclick="openModal('projectModal')">+ New Project</button>
    </div>
    <div class="page-body">
      <div class="tab-bar" style="width:fit-content">
        <button class="tab active" onclick="loadProjects('active',this)">Active</button>
        <button class="tab" onclick="loadProjects('done',this)">Done</button>
        <button class="tab" onclick="loadProjects('all',this)">All</button>
      </div>
      <div id="projectsList"><div class="loading"><div class="spinner"></div></div></div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ NOTES ═══════════════════════════════════════ -->
  <div class="page" id="page-notes">
    <div class="page-header">
      <div>
        <div class="page-title">Quick Notes</div>
        <div class="page-subtitle">Capture ideas, lecture notes, and reminders</div>
      </div>
      <button class="btn btn-primary" onclick="openModal('noteModal')">+ New Note</button>
    </div>
    <div class="page-body">
      <div class="input-group" style="margin-bottom:16px">
        <input type="text" class="input" id="noteSearch" placeholder="Search notes..." oninput="searchNotes(this.value)">
      </div>
      <div id="notesList"><div class="loading"><div class="spinner"></div></div></div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════ AI CHAT ═══════════════════════════════════════ -->
  <div class="page" id="page-ai">
    <div class="page-header" style="padding-bottom:12px">
      <div>
        <div class="page-title">🤖 AI Assistant</div>
        <div class="page-subtitle" id="aiSubtitle">Powered by Groq ⚡ — smart academic planning & automation assistant</div>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="clearChat()">Clear History</button>
    </div>
    <div class="chat-container">
      <div class="ai-quick">
        <button class="ai-quick-btn" onclick="sendQuick('Build me a focused study plan for today based on my timetable and subjects')">📚 Today's study plan</button>
        <button class="ai-quick-btn" onclick="sendQuick('What should I prioritize this week for best results?')">⚡ Weekly priorities</button>
        <button class="ai-quick-btn" onclick="sendQuick('Create a timetable-based revision schedule for my exams')">📅 Exam schedule</button>
        <button class="ai-quick-btn" onclick="sendQuick('Suggest 5 specific tasks for today based on my subjects and classes')">✅ Task suggestions</button>
        <button class="ai-quick-btn" onclick="sendQuick('How can I improve my study consistency and completion rate?')">📈 Study tips</button>
        <button class="ai-quick-btn" onclick="sendQuick('What subjects or COs have I been neglecting? Give me an honest review.')">🔍 Progress review</button>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="chat-msg assistant">
          <div class="chat-bubble">Hey! I'm your DayMark AI powered by Groq ⚡<br><br>I know about your subjects, today's tasks, and your progress. Ask me anything — study plans, prioritization, burnout check, or just "what should I do today?"<br><br>I know your timetable, subjects, COs, tasks, and progress. Ask me to build study plans, revision schedules, or just tell me what day it is and I'll plan it out for you. 🎓</div>
        </div>
      </div>
      <div class="chat-input-area">
        <textarea class="chat-input" id="chatInput" placeholder="Ask anything... (Enter to send, Shift+Enter for newline)" rows="1" onkeydown="handleChatKey(event)"></textarea>
        <button class="chat-send" onclick="sendChat()">➤</button>
      </div>
    </div>
  </div>

</main>

<!-- MOBILE NAV BAR -->
<div class="mobile-bar">
  <div class="mobile-bar-items">
    <button class="mobile-bar-item active" data-page="dashboard"><span class="m-icon">🏠</span>Home</button>
    <button class="mobile-bar-item" data-page="today"><span class="m-icon">✅</span>Today</button>
    <button class="mobile-bar-item" data-page="projects"><span class="m-icon">🚀</span>Projects</button>
    <button class="mobile-bar-item" data-page="progress"><span class="m-icon">📊</span>Stats</button>
    <button class="mobile-bar-item" data-page="ai"><span class="m-icon">🤖</span>AI</button>
  </div>
</div>

<!-- ═══════════════════ MODALS ═══════════════════ -->
<!-- Semester Modal -->
<div class="modal-overlay" id="semModal">
  <div class="modal">
    <div class="modal-header"><div class="modal-title">New Semester</div><button class="modal-close" onclick="closeModal('semModal')">✕</button></div>
    <div class="form-row"><label>Name</label><input class="input" id="semName" placeholder="e.g. Semester 5 — CSE"></div>
    <div class="form-row cols-2">
      <div><label>Start Date</label><input type="date" class="input" id="semStart"></div>
      <div><label>End Date</label><input type="date" class="input" id="semEnd"></div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('semModal')">Cancel</button>
      <button class="btn btn-primary" onclick="createSemester()">Create Semester</button>
    </div>
  </div>
</div>

<!-- Subject Modal -->
<div class="modal-overlay" id="subjectModal">
  <div class="modal">
    <div class="modal-header"><div class="modal-title">Add Subject</div><button class="modal-close" onclick="closeModal('subjectModal')">✕</button></div>
    <div class="form-row">
      <label>Semester</label>
      <select class="select" id="subjectSemSelect"><option value="">-- Select semester --</option></select>
    </div>
    <div class="form-row cols-2">
      <div><label>Subject Code</label><input class="input" id="subjectCode" placeholder="e.g. CS-501"></div>
      <div><label>Credits</label><input type="number" class="input" id="subjectCredits" placeholder="4"></div>
    </div>
    <div class="form-row"><label>Subject Name</label><input class="input" id="subjectName" placeholder="e.g. Operating Systems"></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('subjectModal')">Cancel</button>
      <button class="btn btn-primary" onclick="createSubject()">Add Subject</button>
    </div>
  </div>
</div>

<!-- CO Modal -->
<div class="modal-overlay" id="coModal">
  <div class="modal">
    <div class="modal-header"><div class="modal-title">Add Course Outcome</div><button class="modal-close" onclick="closeModal('coModal')">✕</button></div>
    <input type="hidden" id="coSubjectId">
    <div class="form-row cols-2">
      <div><label>CO Code</label><input class="input" id="coCode" placeholder="CO1"></div>
    </div>
    <div class="form-row"><label>Title</label><input class="input" id="coTitle" placeholder="Understand memory management..."></div>
    <div class="form-row"><label>Description (optional)</label><textarea class="textarea" id="coDesc" placeholder="More detail..."></textarea></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('coModal')">Cancel</button>
      <button class="btn btn-primary" onclick="createCO()">Save CO</button>
    </div>
  </div>
</div>

<!-- Entry Modal -->
<div class="modal-overlay" id="entryModal">
  <div class="modal">
    <div class="modal-header"><div class="modal-title" id="entryModalTitle">Add Task</div><button class="modal-close" onclick="closeModal('entryModal')">✕</button></div>
    <div class="form-row"><label>Text</label><input class="input" id="entryText" placeholder="What needs to be done?"></div>
    <div class="form-row cols-2">
      <div>
        <label>Link to Subject (optional)</label>
        <select class="select" id="entrySubject" onchange="loadCOsForEntry(this.value)"><option value="">None</option></select>
      </div>
      <div>
        <label>Link to CO (optional)</label>
        <select class="select" id="entryCO"><option value="">None</option></select>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('entryModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveEntry()">Add</button>
    </div>
  </div>
</div>

<!-- Project Modal -->
<div class="modal-overlay" id="projectModal">
  <div class="modal">
    <div class="modal-header"><div class="modal-title">New Project</div><button class="modal-close" onclick="closeModal('projectModal')">✕</button></div>
    <div class="form-row"><label>Project Name</label><input class="input" id="projName" placeholder="e.g. Final Year Project"></div>
    <div class="form-row"><label>Description</label><textarea class="textarea" id="projDesc" placeholder="What is this project about?"></textarea></div>
    <div class="form-row cols-2">
      <div>
        <label>Priority</label>
        <select class="select" id="projPriority">
          <option value="high">🔴 High</option>
          <option value="medium" selected>🟡 Medium</option>
          <option value="low">🟢 Low</option>
        </select>
      </div>
      <div><label>Deadline (optional)</label><input type="date" class="input" id="projDeadline"></div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('projectModal')">Cancel</button>
      <button class="btn btn-primary" onclick="createProject()">Create Project</button>
    </div>
  </div>
</div>

<!-- Note Modal -->
<div class="modal-overlay" id="noteModal">
  <div class="modal" style="max-width:640px">
    <div class="modal-header">
      <div class="modal-title" id="noteModalTitle">New Note</div>
      <button class="modal-close" onclick="closeModal('noteModal')">✕</button>
    </div>
    <input type="hidden" id="editNoteId">
    <div class="form-row"><label>Title</label><input class="input" id="noteTitle" placeholder="Note title..."></div>
    <div class="form-row"><label>Content</label><textarea class="textarea" id="noteContent" style="min-height:160px" placeholder="Write anything here..."></textarea></div>
    <div class="form-row"><label>Tags (comma separated)</label><input class="input" id="noteTags" placeholder="study, os, important"></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('noteModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveNote()">Save Note</button>
    </div>
  </div>
</div>

<!-- CO Review Modal (after PDF upload) -->
<div class="modal-overlay" id="coReviewModal">
  <div class="modal" style="max-width:640px">
    <div class="modal-header">
      <div class="modal-title">Review Extracted COs</div>
      <button class="modal-close" onclick="closeModal('coReviewModal')">✕</button>
    </div>
    <div style="font-size:13px;color:var(--text2);margin-bottom:16px" id="coReviewMeta"></div>
    <div id="coReviewList"></div>
    <button class="btn btn-ghost btn-sm" style="margin-bottom:14px" onclick="addCOReviewItem()">+ Add Another CO</button>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('coReviewModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveCOsFromReview()">Save All COs</button>
    </div>
  </div>
</div>



<!-- Edit Semester Modal -->
<div class="modal-overlay" id="editSemModal">
  <div class="modal" style="max-width:420px">
    <div class="modal-header">
      <div class="modal-title">✏️ Edit Semester</div>
      <button class="modal-close" onclick="closeModal('editSemModal')">✕</button>
    </div>
    <input type="hidden" id="editSemId">
    <div><label>Name *</label><input class="input" type="text" id="editSemName"></div>
    <div class="form-row cols-2" style="margin-top:10px">
      <div><label>Start Date *</label><input class="input" type="date" id="editSemStart"></div>
      <div><label>End Date *</label><input class="input" type="date" id="editSemEnd"></div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('editSemModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveSemesterEdit()">Save Changes</button>
    </div>
  </div>
</div>

<!-- Edit Subject Modal -->
<div class="modal-overlay" id="editSubjectModal">
  <div class="modal" style="max-width:420px">
    <div class="modal-header">
      <div class="modal-title">✏️ Edit Subject</div>
      <button class="modal-close" onclick="closeModal('editSubjectModal')">✕</button>
    </div>
    <input type="hidden" id="editSubjectId">
    <div class="form-row cols-2">
      <div><label>Code *</label><input class="input" type="text" id="editSubjectCode"></div>
      <div><label>Credits</label><input class="input" type="number" id="editSubjectCredits" min="1" max="10"></div>
    </div>
    <div style="margin-top:10px"><label>Name *</label><input class="input" type="text" id="editSubjectName"></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('editSubjectModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveSubjectEdit()">Save Changes</button>
    </div>
  </div>
</div>

<!-- Edit CO Modal -->
<div class="modal-overlay" id="editCOModal">
  <div class="modal" style="max-width:480px">
    <div class="modal-header">
      <div class="modal-title">✏️ Edit Course Outcome</div>
      <button class="modal-close" onclick="closeModal('editCOModal')">✕</button>
    </div>
    <input type="hidden" id="editCOId">
    <div class="form-row cols-2">
      <div><label>CO Code *</label><input class="input" type="text" id="editCOCode" placeholder="CO1"></div>
    </div>
    <div style="margin-top:10px"><label>Title *</label><input class="input" type="text" id="editCOTitle"></div>
    <div style="margin-top:10px"><label>Description</label><textarea class="input" id="editCODesc" rows="3" style="resize:vertical"></textarea></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('editCOModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveCOEdit()">Save Changes</button>
    </div>
  </div>
</div>

<!-- Edit Resource Modal -->
<div class="modal-overlay" id="editResourceModal">
  <div class="modal" style="max-width:420px">
    <div class="modal-header">
      <div class="modal-title">✏️ Rename Resource</div>
      <button class="modal-close" onclick="closeModal('editResourceModal')">✕</button>
    </div>
    <input type="hidden" id="editResourceId">
    <div><label>Title</label><input class="input" type="text" id="editResourceTitle"></div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('editResourceModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveResourceEdit()">Save</button>
    </div>
  </div>
</div>

<!-- Add Timetable Slot Modal -->
<div class="modal-overlay" id="addSlotModal">
  <div class="modal" style="max-width:480px">
    <div class="modal-header">
      <div class="modal-title">➕ Add Class Slot</div>
      <button class="modal-close" onclick="closeModal('addSlotModal')">✕</button>
    </div>
    <div class="form-row cols-2">
      <div>
        <label>Day</label>
        <select class="select" id="addSlotDaySelect"></select>
      </div>
      <div>
        <label>Subject Name *</label>
        <input class="input" type="text" id="addSlotSubject" placeholder="e.g. Operating Systems">
      </div>
    </div>
    <div class="form-row cols-2">
      <div>
        <label>Start Time *</label>
        <input class="input" type="time" id="addSlotStart">
      </div>
      <div>
        <label>End Time *</label>
        <input class="input" type="time" id="addSlotEnd">
      </div>
    </div>
    <div class="form-row cols-2">
      <div>
        <label>Subject Code</label>
        <input class="input" type="text" id="addSlotCode" placeholder="e.g. CS501">
      </div>
      <div>
        <label>Room</label>
        <input class="input" type="text" id="addSlotRoom" placeholder="e.g. Room 301">
      </div>
    </div>
    <div>
      <label>Teacher</label>
      <input class="input" type="text" id="addSlotTeacher" placeholder="e.g. Prof. Sharma">
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('addSlotModal')">Cancel</button>
      <button class="btn btn-primary" onclick="saveNewSlot()">Save Slot</button>
    </div>
  </div>
</div>

<!-- Subject Import from PDF Modal -->
<div class="modal-overlay" id="subjectImportModal">
  <div class="modal" style="max-width:480px">
    <div class="modal-header">
      <div class="modal-title">🤖 Import Subject from PDF</div>
      <button class="modal-close" onclick="closeModal('subjectImportModal')">✕</button>
    </div>
    <div style="background:rgba(124,110,245,0.08);border-radius:10px;padding:12px;margin-bottom:16px;font-size:13px;color:var(--text2)">
      Upload any syllabus PDF → Groq AI automatically extracts the subject name, code, credits, and all course outcomes. No manual entry needed.
    </div>
    <div>
      <label>Select Semester *</label>
      <select class="select" id="importSubjSemSelect"></select>
    </div>
    <div style="margin-top:12px">
      <label>Syllabus PDF *</label>
      <input type="file" accept=".pdf" id="importSubjFile" style="margin-top:6px;width:100%;padding:8px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:13px">
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal('subjectImportModal')">Cancel</button>
      <button class="btn btn-primary" id="importSubjBtn" onclick="importSubjectFromPdf()">📤 Import</button>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════
const S = {
  currentPage: 'dashboard',
  currentDate: new Date().toISOString().split('T')[0],
  dayData: null,
  habits: [],
  tasks: [],
  allSubjects: [],
  allSemesters: [],
  chartInstance: null,
  editingNoteId: null,
  coReviewSubjectId: null,
  currentSemFilter: null,
};

// ═══════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════
const API = {
  base: '',
  async get(path) {
    const r = await fetch(this.base + path);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(this.base + path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async patch(path, body) {
    const r = await fetch(this.base + path, { method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async delete(path) {
    const r = await fetch(this.base + path, { method: 'DELETE' });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};

// ═══════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════
function toast(msg, type='info') {
  const icons = { success:'✅', error:'❌', info:'ℹ️', warning:'⚠️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span class="toast-icon">${icons[type]}</span><span class="toast-msg">${msg}</span><button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  document.getElementById('toastContainer').appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════
function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item, .mobile-bar-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelectorAll(`[data-page="${page}"]`).forEach(n => n.classList.add('active'));
  S.currentPage = page;
  loadPage(page);
}

document.querySelectorAll('.nav-item[data-page], .mobile-bar-item[data-page]').forEach(btn => {
  btn.addEventListener('click', () => navigate(btn.dataset.page));
});

function loadPage(page) {
  const loaders = {
    dashboard: loadDashboard,
    today: () => loadDay(S.currentDate),
    progress: loadProgress,
    semesters: () => loadSemesters('active'),
    subjects: loadSubjectsPage,
    resources: loadResourcesPage,
    projects: () => loadProjects('active'),
    notes: loadNotes,
    ai: loadChatHistory,
    timetable: loadTimetable,
  };
  if (loaders[page]) loaders[page]();
}

// ═══════════════════════════════════════════════════════════
// MODALS
// ═══════════════════════════════════════════════════════════
function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }
document.querySelectorAll('.modal-overlay').forEach(m => {
  m.addEventListener('click', e => { if (e.target === m) m.classList.remove('open'); });
});

// ═══════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════
async function loadDashboard() {
  const now = new Date();
  const h = now.getHours();
  const greeting = h < 12 ? 'Good Morning' : h < 17 ? 'Good Afternoon' : 'Good Evening';
  document.getElementById('dashGreeting').textContent = `${greeting}! 👋`;
  document.getElementById('dashDate').textContent = now.toLocaleDateString('en-US', {weekday:'long', year:'numeric', month:'long', day:'numeric'});

  try {
    const s = await API.get('/api/dashboard/summary');
    document.getElementById('dStreak').textContent = s.streak;
    document.getElementById('sidebarStreak').textContent = s.streak;
    document.getElementById('dMonth').textContent = s.month_completion + '%';
    document.getElementById('dSubs').textContent = s.active_subjects;
    document.getElementById('dWeek').textContent = s.week_tasks_done;

    // Groq AI status badge
    const groqDot = document.getElementById('groqDot');
    const groqStatus = document.getElementById('groqStatus');
    if (s.groq_enabled) {
      groqDot.style.background = 'var(--green)';
      groqStatus.textContent = '⚡ Groq: ' + (s.groq_model || 'active');
    } else {
      groqDot.style.background = 'var(--yellow)';
      groqStatus.textContent = 'Set GROQ_API_KEY';
    }

    // Today snapshot
    const t = s.today;
    const totalToday = t.habits + t.tasks;
    const doneToday = t.done_habits + t.done_tasks;
    const pct = totalToday > 0 ? Math.round(doneToday / totalToday * 100) : 0;
    document.getElementById('dashTodaySnap').innerHTML = `
      <div style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <span style="font-size:14px;color:var(--text2)">${doneToday}/${totalToday} items done</span>
          <span style="font-weight:700;color:${pct>=70?'var(--green)':pct>=40?'var(--yellow)':'var(--red)'}">${pct}%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill ${pct>=70?'progress-green':pct>=40?'progress-yellow':'progress-red'}" style="width:${pct}%"></div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="background:var(--bg3);border-radius:10px;padding:10px;text-align:center">
          <div style="font-size:22px;font-weight:800;font-family:'Syne',sans-serif">${t.done_habits}/${t.habits}</div>
          <div style="font-size:12px;color:var(--text2)">Habits</div>
        </div>
        <div style="background:var(--bg3);border-radius:10px;padding:10px;text-align:center">
          <div style="font-size:22px;font-weight:800;font-family:'Syne',sans-serif">${t.done_tasks}/${t.tasks}</div>
          <div style="font-size:12px;color:var(--text2)">Tasks</div>
        </div>
      </div>
      <div style="margin-top:10px">
        <span class="badge ${t.day_locked ? 'badge-green' : 'badge-yellow'}">${t.day_locked ? '🔒 Locked' : '🔓 In Progress'}</span>
      </div>`;
  } catch(e) { console.error(e); }

  // Load today's timetable in dashboard
  try {
    const allSlots = await API.get('/api/timetable');
    const dow = (new Date().getDay() + 6) % 7;
    const todaySlots = allSlots.filter(s => s.day_of_week === dow)
      .sort((a,b) => a.time_start.localeCompare(b.time_start));
    const el = document.getElementById('dashTimetable');
    if (todaySlots.length) {
      el.innerHTML = todaySlots.map(s => `
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)">
          <div style="background:var(--accent);color:white;border-radius:6px;padding:3px 7px;font-size:11px;font-weight:600">${s.time_start}</div>
          <div style="font-size:13px;font-weight:500">${s.subject_name}</div>
          ${s.room ? `<div style="font-size:12px;color:var(--text2);margin-left:auto">📍${s.room}</div>` : ''}
        </div>`).join('');
    } else {
      el.innerHTML = `<div style="color:var(--text3);font-size:13px;padding:4px 0">No classes today. <span style="cursor:pointer;color:var(--accent)" onclick="navigate('timetable')">Add timetable →</span></div>`;
    }
  } catch(e) {}

  loadHeatmap();
  loadDashboardProjects();
}

async function loadHeatmap() {
  const today = new Date();
  const weeks = 12;
  const startDay = new Date(today);
  startDay.setDate(startDay.getDate() - (weeks * 7) + 1);

  try {
    const data = await API.get('/api/progress/performance-chart?period=all');
    const dayMap = {};
    data.data.forEach(d => { dayMap[d.date] = d.completion_percentage; });

    let rows = Array.from({length:7}, () => []);
    let curr = new Date(startDay);
    let dayOfWeek = curr.getDay();
    for (let i = 0; i < dayOfWeek; i++) rows[i].push({empty:true});

    while (curr <= today) {
      const iso = curr.toISOString().split('T')[0];
      const pct = dayMap[iso] || 0;
      const isToday = iso === today.toISOString().split('T')[0];
      rows[curr.getDay()].push({date: iso, pct, isToday});
      curr.setDate(curr.getDate() + 1);
    }

    const numWeeks = Math.ceil((today - startDay) / (7 * 24 * 3600 * 1000)) + 1;
    let html = '<div style="display:flex;gap:3px">';
    for (let w = 0; w < numWeeks; w++) {
      html += '<div style="display:flex;flex-direction:column;gap:3px">';
      for (let d = 0; d < 7; d++) {
        const cell = rows[d][w];
        if (!cell || cell.empty) { html += '<div style="width:14px;height:14px"></div>'; continue; }
        const lvl = cell.pct >= 80 ? 'l4' : cell.pct >= 50 ? 'l3' : cell.pct >= 20 ? 'l2' : cell.pct > 0 ? 'l1' : '';
        html += `<div class="heatmap-cell ${lvl} ${cell.isToday?'today-cell':''}" title="${cell.date}: ${Math.round(cell.pct)}%"></div>`;
      }
      html += '</div>';
    }
    html += '</div>';
    document.getElementById('heatmapContainer').innerHTML = html;
  } catch(e) {
    document.getElementById('heatmapContainer').innerHTML = '<div style="color:var(--text3);font-size:13px">No data yet</div>';
  }
}

async function loadDashboardProjects() {
  try {
    const projs = await API.get('/api/projects?status=active');
    const el = document.getElementById('dashProjects');
    if (!projs.length) { el.innerHTML = '<div class="empty-state" style="padding:20px"><div class="empty-icon">🚀</div><div class="empty-desc">No active projects. <a href="#" onclick="navigate(\'projects\');return false">Create one →</a></div></div>'; return; }
    el.innerHTML = projs.slice(0,4).map(p => `
      <div style="padding:10px 0;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px">
        <div style="flex:1">
          <div style="font-weight:500;margin-bottom:4px">${p.name}</div>
          <div class="progress-bar"><div class="progress-fill progress-purple" style="width:${p.progress}%"></div></div>
        </div>
        <span style="font-size:13px;color:var(--text2)">${p.progress}%</span>
        <span class="badge badge-${p.priority==='high'?'red':p.priority==='medium'?'yellow':'green'}">${p.priority}</span>
      </div>`).join('');
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════
// TIMETABLE PAGE
// ═══════════════════════════════════════════════════════════
const DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];

async function loadTimetable() {
  try {
    const slots = await API.get('/api/timetable');
    const el = document.getElementById('weeklyTimetable');
    if (!slots.length) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">🗓️</div><div class="empty-title">No timetable yet</div><div class="empty-desc">Upload your routine PDF above or click "+ Add Slot" to build your schedule manually.</div></div>`;
      return;
    }
    // Group by day
    const grouped = {};
    for (let d=0;d<7;d++) grouped[d]=[];
    slots.forEach(s => grouped[s.day_of_week].push(s));
    Object.keys(grouped).forEach(d => grouped[d].sort((a,b)=>a.time_start.localeCompare(b.time_start)));

    let html = '<div style="display:grid;gap:16px">';
    for (let d=0;d<7;d++) {
      const daySlots = grouped[d];
      if (!daySlots.length) continue;
      html += `<div>
        <div style="font-weight:700;font-size:13px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--border)">${DAYS[d]}</div>
        <div style="display:grid;gap:6px">
          ${daySlots.map(s => `
          <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;background:var(--bg3);border-radius:10px;border-left:3px solid var(--accent)">
            <div style="background:var(--accent);color:white;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:600;white-space:nowrap">${s.time_start}–${s.time_end}</div>
            <div style="flex:1">
              <div style="font-weight:600;font-size:14px">${s.subject_name} ${s.subject_code ? `<span style="font-size:12px;color:var(--text2)">(${s.subject_code})</span>` : ''}</div>
              ${s.teacher ? `<div style="font-size:12px;color:var(--text2)">👤 ${s.teacher}</div>` : ''}
              ${s.room ? `<div style="font-size:12px;color:var(--text2)">📍 ${s.room}</div>` : ''}
            </div>
            <button class="btn btn-xs btn-danger" onclick="deleteTimetableSlot(${s.id})">✕</button>
          </div>`).join('')}
        </div>
      </div>`;
    }
    html += '</div>';
    el.innerHTML = html;
  } catch(e) { toast('Error loading timetable', 'error'); }
}

async function importRoutinePdf(input) {
  const file = input.files[0];
  if (!file) return;
  const zone = document.getElementById('timetableImportZone');
  const prog = document.getElementById('timetableImportProgress');
  const msg  = document.getElementById('timetableImportMsg');
  prog.style.display = 'block';
  msg.textContent = '⚡ Groq AI is reading your routine PDF...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/timetable/import-pdf', {method:'POST', body:fd});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Import failed');
    prog.style.display = 'none';
    input.value = '';
    if (data.imported > 0) {
      toast(`✅ Imported ${data.imported} class slots from your routine!`, 'success');
      loadTimetable();
    } else {
      toast('Could not parse timetable. Try a different PDF.', 'error');
    }
  } catch(e) {
    prog.style.display = 'none';
    input.value = '';
    toast('Import failed: ' + e.message, 'error');
  }
}

async function deleteTimetableSlot(id) {
  if (!confirm('Remove this class slot?')) return;
  try {
    await API.delete(`/api/timetable/slots/${id}`);
    loadTimetable();
    toast('Slot removed', 'info');
  } catch(e) { toast('Error', 'error'); }
}

async function clearTimetable() {
  if (!confirm('Clear ALL timetable slots? This cannot be undone.')) return;
  try {
    await API.delete('/api/timetable/clear');
    loadTimetable();
    toast('Timetable cleared', 'info');
  } catch(e) { toast('Error', 'error'); }
}

function openAddSlotModal() {
  const days = DAYS.map((d,i) => `<option value="${i}">${d}</option>`).join('');
  const modal = document.getElementById('addSlotModal');
  document.getElementById('addSlotDaySelect').innerHTML = days;
  openModal('addSlotModal');
}

async function saveNewSlot() {
  const day = parseInt(document.getElementById('addSlotDaySelect').value);
  const start = document.getElementById('addSlotStart').value;
  const end   = document.getElementById('addSlotEnd').value;
  const subj  = document.getElementById('addSlotSubject').value.trim();
  const code  = document.getElementById('addSlotCode').value.trim();
  const room  = document.getElementById('addSlotRoom').value.trim();
  const teacher = document.getElementById('addSlotTeacher').value.trim();
  if (!subj || !start || !end) { toast('Subject, start and end time are required', 'error'); return; }
  try {
    await API.post('/api/timetable/slots', {day_of_week:day, time_start:start, time_end:end,
      subject_name:subj, subject_code:code, room, teacher});
    closeModal('addSlotModal');
    loadTimetable();
    toast('✅ Class slot added!', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════════════════════════
// TODAY PAGE
// ═══════════════════════════════════════════════════════════
function changeDay(delta) {
  const d = new Date(S.currentDate);
  d.setDate(d.getDate() + delta);
  S.currentDate = d.toISOString().split('T')[0];
  loadDay(S.currentDate);
}

function goToday() {
  S.currentDate = new Date().toISOString().split('T')[0];
  loadDay(S.currentDate);
}

async function loadDay(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  const today = new Date().toISOString().split('T')[0];
  const isToday = dateStr === today;
  document.getElementById('todayDateDisplay').textContent = isToday ? '📅 Today' :
    d.toLocaleDateString('en-US', {weekday:'short', month:'short', day:'numeric'});

  try {
    const data = await API.get(`/api/day/${dateStr}`);
    S.dayData = data;
    S.habits = data.habits || [];
    S.tasks = data.tasks || [];

    const locked = data.locked;
    document.getElementById('lockedBanner').style.display = locked ? 'block' : 'none';
    document.getElementById('addHabitBtn').disabled = locked;
    document.getElementById('addTaskBtn').disabled = locked;
    document.getElementById('endDayBtn').style.display = locked ? 'none' : 'inline-flex';
    document.getElementById('aiSuggestBtn').style.display = isToday ? 'inline-flex' : 'none';

    renderHabits();
    renderTasks();
    renderTodayProgress();
  } catch(e) {
    toast('Error loading day: ' + e.message, 'error');
  }

  // Load today's class schedule
  try {
    const dow = new Date(dateStr + 'T00:00:00').getDay();
    // Convert JS Sunday=0 to Python Monday=0
    const pythonDow = (dow + 6) % 7;
    const allSlots = await API.get('/api/timetable');
    const todaySlots = allSlots.filter(s => s.day_of_week === pythonDow)
      .sort((a,b) => a.time_start.localeCompare(b.time_start));
    const el = document.getElementById('todayClassesList');
    if (todaySlots.length) {
      el.innerHTML = todaySlots.map(s => `
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
          <div style="background:var(--accent);color:white;border-radius:8px;padding:4px 8px;font-size:12px;font-weight:600;white-space:nowrap">${s.time_start}-${s.time_end}</div>
          <div style="flex:1">
            <div style="font-weight:500;font-size:14px">${s.subject_name}</div>
            ${s.room ? `<div style="font-size:12px;color:var(--text2)">📍 ${s.room}</div>` : ''}
          </div>
          ${s.subject_code ? `<span class="badge badge-purple">${s.subject_code}</span>` : ''}
        </div>`).join('');
    } else {
      el.innerHTML = `<div style="color:var(--text3);font-size:13px">No classes today. <span style="cursor:pointer;color:var(--accent)" onclick="navigate('timetable')">Set up timetable →</span></div>`;
    }
  } catch(e) {}

  if (dateStr === today) {
    try { await API.post('/api/progress/check-and-reset-yesterday', {}); } catch {}
  }
}

function renderHabits() {
  const el = document.getElementById('habitsList');
  const locked = S.dayData?.locked;
  if (!S.habits.length) {
    el.innerHTML = `<div class="empty-state" style="padding:20px"><div class="empty-icon">🌅</div><div class="empty-desc">${locked?'No habits logged':'Add your daily habits'}</div></div>`;
    return;
  }
  el.innerHTML = S.habits.map((h,i) => `
    <div class="task-item ${h.done?'done':''}" id="habit-${i}">
      <div class="task-cb habit-cb ${h.done?'checked':''}" onclick="toggleEntry('habit',${i})"></div>
      <div style="flex:1">
        <div class="task-text">${escHtml(h.text)}</div>
        ${h.subject_id ? `<div class="task-meta">📖 Linked to subject</div>` : ''}
      </div>
      <div class="task-actions">
        ${!locked ? `<button class="btn btn-xs btn-danger" onclick="removeEntry('habit',${i})">✕</button>` : ''}
      </div>
    </div>`).join('');
}

function renderTasks() {
  const el = document.getElementById('tasksList');
  const locked = S.dayData?.locked;
  if (!S.tasks.length) {
    el.innerHTML = `<div class="empty-state" style="padding:20px"><div class="empty-icon">📋</div><div class="empty-desc">${locked?'No tasks logged':'Add today\'s tasks'}</div></div>`;
    return;
  }
  el.innerHTML = S.tasks.map((t,i) => `
    <div class="task-item ${t.done?'done':''}" id="task-${i}">
      <div class="task-cb ${t.done?'checked':''}" onclick="toggleEntry('task',${i})"></div>
      <div style="flex:1">
        <div class="task-text">${escHtml(t.text)}</div>
        ${t.subject_id ? `<div class="task-meta">📖 Linked to subject</div>` : ''}
      </div>
      <div class="task-actions">
        ${!locked ? `<button class="btn btn-xs btn-danger" onclick="removeEntry('task',${i})">✕</button>` : ''}
      </div>
    </div>`).join('');
}

function renderTodayProgress() {
  const total = S.habits.length + S.tasks.length;
  const done = S.habits.filter(h=>h.done).length + S.tasks.filter(t=>t.done).length;
  const pct = total > 0 ? Math.round(done/total*100) : 0;
  document.getElementById('todayProgressBar').innerHTML = `
    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
      <span style="color:var(--text2)">${done}/${total} items</span>
      <span style="font-weight:700;color:${pct>=70?'var(--green)':pct>=40?'var(--yellow)':'var(--red)'}">${pct}%</span>
    </div>
    <div class="progress-bar" style="height:10px"><div class="progress-fill ${pct>=70?'progress-green':pct>=40?'progress-yellow':'progress-red'}" style="width:${pct}%"></div></div>`;
}

async function toggleEntry(type, idx) {
  if (S.dayData?.locked) return;
  const arr = type === 'habit' ? S.habits : S.tasks;
  arr[idx].done = !arr[idx].done;
  if (type === 'habit') renderHabits(); else renderTasks();
  renderTodayProgress();
  await syncDay();
}

async function removeEntry(type, idx) {
  if (S.dayData?.locked) return;
  if (type === 'habit') S.habits.splice(idx, 1); else S.tasks.splice(idx, 1);
  if (type === 'habit') renderHabits(); else renderTasks();
  renderTodayProgress();
  await syncDay();
}

let currentEntryType = 'task';
async function addEntry(type) {
  currentEntryType = type;
  document.getElementById('entryModalTitle').textContent = type === 'habit' ? 'Add Habit' : 'Add Task';
  document.getElementById('entryText').value = '';
  document.getElementById('entrySubject').value = '';
  document.getElementById('entryCO').innerHTML = '<option value="">None</option>';

  const sel = document.getElementById('entrySubject');
  sel.innerHTML = '<option value="">None</option>';
  try {
    const subs = await API.get('/api/subjects/all');
    S.allSubjects = subs;
    subs.forEach(s => sel.innerHTML += `<option value="${s.id}">${s.code} — ${s.name}</option>`);
  } catch {}

  openModal('entryModal');
  setTimeout(() => document.getElementById('entryText').focus(), 100);
}

async function loadCOsForEntry(subjectId) {
  const sel = document.getElementById('entryCO');
  sel.innerHTML = '<option value="">None</option>';
  if (!subjectId) return;
  try {
    const cos = await API.get(`/api/subjects/${subjectId}/course-outcomes`);
    cos.forEach(c => sel.innerHTML += `<option value="${c.id}">${c.co_code}: ${c.title.substring(0,60)}</option>`);
  } catch {}
}

async function saveEntry() {
  const text = document.getElementById('entryText').value.trim();
  if (!text) { toast('Please enter some text', 'warning'); return; }
  const subjectId = document.getElementById('entrySubject').value || null;
  const coId = document.getElementById('entryCO').value || null;
  const entry = { entry_type: currentEntryType, text, done: false,
    subject_id: subjectId ? parseInt(subjectId) : null,
    co_id: coId ? parseInt(coId) : null };
  if (currentEntryType === 'habit') S.habits.push(entry);
  else S.tasks.push(entry);
  closeModal('entryModal');
  if (currentEntryType === 'habit') renderHabits(); else renderTasks();
  renderTodayProgress();
  await syncDay();
  toast(`${currentEntryType === 'habit' ? 'Habit' : 'Task'} added!`, 'success');
}

async function syncDay() {
  try {
    await API.post(`/api/day/${S.currentDate}/sync`, { habits: S.habits, tasks: S.tasks });
  } catch(e) { toast('Sync error: ' + e.message, 'error'); }
}

async function endDay() {
  const total = S.habits.length + S.tasks.length;
  const done = S.habits.filter(h=>h.done).length + S.tasks.filter(t=>t.done).length;
  if (total > 0 && done === 0) {
    if (!confirm('You have 0 tasks done. Lock the day anyway?')) return;
  }
  try {
    await syncDay();
    await API.post(`/api/day/${S.currentDate}/end`, {});
    S.dayData = {...S.dayData, locked: true};
    document.getElementById('lockedBanner').style.display = 'block';
    document.getElementById('endDayBtn').style.display = 'none';
    document.getElementById('addHabitBtn').disabled = true;
    document.getElementById('addTaskBtn').disabled = true;
    renderHabits(); renderTasks();
    toast('🔒 Day locked! Great work.', 'success');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function unlockDay() {
  if (!confirm('Unlock this day to make changes?')) return;
  try {
    await API.post(`/api/day/${S.currentDate}/unlock`, {});
    loadDay(S.currentDate);
    toast('Day unlocked', 'info');
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function generateFromTimetable() {
  try {
    const r = await API.post(`/api/day/${S.currentDate}/generate-from-timetable`, {});
    if (r.added > 0) {
      toast(`🗓️ Added ${r.added} classes to your schedule!`, 'success');
      await loadDay(S.currentDate);
    } else {
      toast(r.message || 'No timetable slots for today.', 'info');
    }
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function loadAISuggestions() {
  document.getElementById('aiSuggestionsBox').style.display = 'block';
  document.getElementById('aiSuggestContent').innerHTML = '<div class="loading"><div class="spinner"></div> ⚡ Groq is thinking...</div>';
  try {
    const data = await API.get(`/api/day/${S.currentDate}/ai-suggestions`);
    const habits = data.habits || [];
    const tasks = data.tasks || [];
    document.getElementById('aiSuggestContent').innerHTML = `
      <div style="margin-bottom:10px">
        <div style="font-size:12px;color:var(--text2);margin-bottom:6px">HABITS</div>
        <div>${habits.map(h => `<span class="suggestion-pill" onclick="addSuggestion('habit','${escHtml(h)}')">${h} <span class="pill-add">+</span></span>`).join('')}</div>
      </div>
      <div>
        <div style="font-size:12px;color:var(--text2);margin-bottom:6px">TASKS</div>
        <div>${tasks.map(t => `<span class="suggestion-pill" onclick="addSuggestion('task','${escHtml(t)}')">${t} <span class="pill-add">+</span></span>`).join('')}</div>
      </div>`;
  } catch(e) {
    document.getElementById('aiSuggestContent').innerHTML = '<div style="color:var(--text2);font-size:13px">Could not load suggestions. Make sure GROQ_API_KEY is set in your .env file.</div>';
  }
}

async function addSuggestion(type, text) {
  const entry = { entry_type: type, text, done: false, subject_id: null, co_id: null };
  if (type === 'habit') S.habits.push(entry); else S.tasks.push(entry);
  if (type === 'habit') renderHabits(); else renderTasks();
  renderTodayProgress();
  await syncDay();
  toast(`Added: ${text}`, 'success');
}

// ═══════════════════════════════════════════════════════════
// PROGRESS
// ═══════════════════════════════════════════════════════════
async function loadProgress() {
  await loadChart('7d');
  await loadAnalysis();
  await renderCalendar();
  try {
    const s = await API.get('/api/progress/streak');
    const analysis = await API.get('/api/progress/analysis');
    const m = analysis.metrics || {};
    document.getElementById('progressStats').innerHTML = `
      <div class="stat-card"><div class="stat-icon">🔥</div><div class="stat-value">${s.streak}</div><div class="stat-label">Current Streak</div></div>
      <div class="stat-card"><div class="stat-icon">✅</div><div class="stat-value">${m.avg_completion_rate||0}%</div><div class="stat-label">Avg Completion</div></div>
      <div class="stat-card"><div class="stat-icon">📅</div><div class="stat-value">${m.completed_days_30d||0}</div><div class="stat-label">Days Locked (30d)</div></div>
      <div class="stat-card"><div class="stat-icon">📊</div><div class="stat-value">${m.avg_daily_tasks||0}</div><div class="stat-label">Avg Tasks/Day</div></div>`;
  } catch(e) {}
}

async function loadChart(period, btn) {
  if (btn) {
    document.querySelectorAll('.tab-bar .tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
  }
  try {
    const data = await API.get(`/api/progress/performance-chart?period=${period}`);
    const labels = data.data.map(d => d.label);
    const comps = data.data.map(d => d.completion_percentage);
    const totals = data.data.map(d => d.total_tasks);

    const ctx = document.getElementById('progressChart');
    if (S.chartInstance) S.chartInstance.destroy();
    S.chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Completion %', data: comps, backgroundColor: comps.map(v => v >= 70 ? 'rgba(34,211,160,0.7)' : v >= 40 ? 'rgba(245,200,66,0.7)' : 'rgba(245,94,94,0.7)'), borderRadius: 6, yAxisID: 'y' },
          { label: 'Total Tasks', data: totals, type: 'line', borderColor: 'rgba(124,110,245,0.6)', backgroundColor: 'rgba(124,110,245,0.1)', tension: 0.4, fill: true, pointRadius: 3, yAxisID: 'y1' }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#9090a8', font: { size: 12 } } } },
        scales: {
          x: { ticks: { color: '#5a5a72', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { min: 0, max: 100, ticks: { color: '#5a5a72', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: '%', color: '#5a5a72' } },
          y1: { position: 'right', ticks: { color: '#5a5a72', font: { size: 11 } }, grid: { display: false } }
        }
      }
    });
  } catch(e) {}
}

async function loadAnalysis() {
  try {
    const data = await API.get('/api/progress/analysis');
    const healthColor = data.status === 'stable' ? 'var(--green)' : data.status === 'fragile' ? 'var(--yellow)' : 'var(--red)';
    const m = data.metrics;
    document.getElementById('analysisContent').innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <span style="font-size:24px">${data.status === 'stable' ? '💪' : data.status === 'fragile' ? '⚠️' : '🚨'}</span>
        <div>
          <div style="font-weight:700;color:${healthColor}">${data.status.toUpperCase()}</div>
          <div style="font-size:12px;color:var(--text2)">System health</div>
        </div>
      </div>
      ${m ? `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        <div style="background:var(--bg3);border-radius:10px;padding:12px;text-align:center">
          <div style="font-size:22px;font-weight:800">${m.avg_completion_rate}%</div>
          <div style="font-size:11px;color:var(--text2)">Avg Completion</div>
        </div>
        <div style="background:var(--bg3);border-radius:10px;padding:12px;text-align:center">
          <div style="font-size:22px;font-weight:800">${m.avg_daily_tasks}</div>
          <div style="font-size:11px;color:var(--text2)">Avg Tasks/Day</div>
        </div>
      </div>` : ''}
      ${data.warnings.length ? `<div style="margin-bottom:10px"><div style="font-size:12px;font-weight:600;color:var(--red);margin-bottom:6px">⚠️ Warnings</div>${data.warnings.map(w=>`<div style="font-size:13px;color:var(--text2);padding:4px 0">• ${w}</div>`).join('')}</div>` : ''}
      <div><div style="font-size:12px;font-weight:600;color:var(--green);margin-bottom:6px">💡 Recommendations</div>${data.recommendations.map(r=>`<div style="font-size:13px;color:var(--text2);padding:4px 0">→ ${r}</div>`).join('')}</div>`;
  } catch(e) { document.getElementById('analysisContent').innerHTML = '<div style="color:var(--text2);font-size:13px">No data yet</div>'; }
}

async function renderCalendar() {
  const today = new Date();
  const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0);

  try {
    const data = await API.get(`/api/progress/performance-chart?period=1m`);
    const dayMap = {};
    data.data.forEach(d => dayMap[d.date] = { pct: d.completion_percentage, total: d.total_tasks });

    let html = `<div style="font-weight:600;margin-bottom:10px;font-size:14px">${today.toLocaleDateString('en-US',{month:'long',year:'numeric'})}</div>`;
    html += '<div class="cal-header">';
    ['S','M','T','W','T','F','S'].forEach(d => html += `<span>${d}</span>`);
    html += '</div><div class="calendar-grid">';

    for (let i = 0; i < firstDay.getDay(); i++) html += '<div class="cal-day empty"></div>';
    for (let d = 1; d <= lastDay.getDate(); d++) {
      const dateStr = `${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
      const isToday = d === today.getDate();
      const dayInfo = dayMap[dateStr];
      const isFuture = d > today.getDate();
      let cls = 'cal-day';
      if (isToday) cls += ' today';
      else if (isFuture) cls += ' future';
      else if (dayInfo && dayInfo.total > 0 && dayInfo.pct >= 50) cls += ' locked';
      else if (!isFuture) cls += ' missed';
      html += `<div class="${cls}" title="${dateStr}: ${dayInfo ? Math.round(dayInfo.pct)+'%' : 'no data'}"><span class="day-num">${d}</span></div>`;
    }
    html += '</div>';
    document.getElementById('calendarView').innerHTML = html;
  } catch(e) {}
}

// ═══════════════════════════════════════════════════════════
// SEMESTERS
// ═══════════════════════════════════════════════════════════
async function loadSemesters(status = 'active', btn) {
  if (btn) {
    btn.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
  }
  const el = document.getElementById('semestersList');
  el.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const sems = await API.get(`/api/semesters?status=${status}`);
    S.allSemesters = sems;
    if (!sems.length) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">🎓</div><div class="empty-title">No ${status} semesters</div><div class="empty-desc">Create your first semester to get started</div></div>`;
      return;
    }
    el.innerHTML = sems.map(s => `
      <div class="card" style="margin-bottom:12px">
        <div style="display:flex;align-items:flex-start;justify-content:space-between">
          <div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:16px">${escHtml(s.name)}</div>
            <div style="font-size:13px;color:var(--text2);margin-top:4px">${fmtDate(s.start_date)} — ${fmtDate(s.end_date)}</div>
          </div>
          <div style="display:flex;gap:8px;align-items:center">
            <span class="badge badge-${s.status === 'active' ? 'green' : s.status === 'archived' ? 'blue' : 'yellow'}">${s.status}</span>
            ${s.status !== 'active' ? `<button class="btn btn-ghost btn-sm" onclick="restoreSemester(${s.id})">↩ Restore</button>` : ''}
            <button class="btn btn-ghost btn-sm" onclick="openEditSemester(${s.id},'${s.name}','${s.start_date}','${s.end_date}')">✏️ Edit</button>
            <button class="btn btn-ghost btn-sm" onclick="deleteSemester(${s.id},'${s.name}')">🗑️ Delete</button>
          </div>
        </div>
      </div>`).join('');
  } catch(e) { el.innerHTML = `<div style="color:var(--red);padding:20px">${e.message}</div>`; }
}

async function createSemester() {
  const name = document.getElementById('semName').value.trim();
  const start = document.getElementById('semStart').value;
  const end = document.getElementById('semEnd').value;
  if (!name || !start || !end) { toast('Fill all fields', 'warning'); return; }
  try {
    await API.post('/api/semesters', { name, start_date: start, end_date: end });
    closeModal('semModal');
    document.getElementById('semName').value = '';
    document.getElementById('semStart').value = '';
    document.getElementById('semEnd').value = '';
    toast('Semester created!', 'success');
    loadSemesters();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteSemester(id, name) {
  const mode = prompt(`Delete "${name}"?\n\nType:\n  soft — hide from view (reversible)\n  archive — archive it (reversible)\n  hard — PERMANENT delete\n\nEnter mode:`);
  if (!mode || !['soft','archive','hard'].includes(mode)) return;
  if (mode === 'hard' && !confirm('Are you sure? This is PERMANENT.')) return;
  try {
    await API.delete(`/api/semesters/${id}?mode=${mode}`);
    toast(`Semester ${mode === 'hard' ? 'deleted' : mode === 'archive' ? 'archived' : 'hidden'}`, 'success');
    loadSemesters();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function restoreSemester(id) {
  try {
    await API.post(`/api/semesters/${id}/restore`, {});
    toast('Semester restored!', 'success');
    loadSemesters();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════════════════════════
// SUBJECTS
// ═══════════════════════════════════════════════════════════
function openSubjectImportModal() {
  // Populate semester selector
  API.get('/api/semesters').then(sems => {
    const sel = document.getElementById('importSubjSemSelect');
    sel.innerHTML = sems.map(s => `<option value="${s.id}">${s.name}</option>`).join('') || '<option value="">No semesters</option>';
  });
  openModal('subjectImportModal');
}

async function importSubjectFromPdf() {
  const semId = document.getElementById('importSubjSemSelect').value;
  const fileInput = document.getElementById('importSubjFile');
  const file = fileInput.files[0];
  if (!semId) { toast('Select a semester first', 'error'); return; }
  if (!file) { toast('Choose a PDF file first', 'error'); return; }
  const btn = document.getElementById('importSubjBtn');
  btn.disabled = true; btn.textContent = '⚡ Groq AI is reading...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch(`/api/subjects/import-from-pdf?semester_id=${semId}`, {method:'POST', body:fd});
    const data = await r.json();
    btn.disabled = false; btn.textContent = '📤 Import';
    fileInput.value = '';
    if (!r.ok) throw new Error(data.detail || 'Import failed');
    if (data.success) {
      closeModal('subjectImportModal');
      toast(`✅ ${data.message}`, 'success');
      loadSubjectsPage();
    } else {
      toast(data.message || 'Import failed', 'error');
    }
  } catch(e) {
    btn.disabled = false; btn.textContent = '📤 Import';
    toast('Import failed: ' + e.message, 'error');
  }
}

async function loadSubjectsPage() {
  try {
    const sems = await API.get('/api/semesters?status=active');
    S.allSemesters = sems;
    const filterEl = document.getElementById('semFilterChips');
    filterEl.innerHTML = `<div class="chip ${!S.currentSemFilter?'selected':''}" onclick="filterBySem(null,this)">All</div>` +
      sems.map(s => `<div class="chip ${S.currentSemFilter===s.id?'selected':''}" onclick="filterBySem(${s.id},this)">${escHtml(s.name)}</div>`).join('');

    const sel = document.getElementById('subjectSemSelect');
    sel.innerHTML = '<option value="">-- Select semester --</option>';
    sems.forEach(s => sel.innerHTML += `<option value="${s.id}">${escHtml(s.name)}</option>`);

    await loadSubjectsList();
  } catch(e) {}
}

async function filterBySem(semId, el) {
  S.currentSemFilter = semId;
  document.querySelectorAll('#semFilterChips .chip').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  await loadSubjectsList();
}

async function loadSubjectsList() {
  const el = document.getElementById('subjectsList');
  el.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    let sems = S.currentSemFilter ? S.allSemesters.filter(s => s.id === S.currentSemFilter) : S.allSemesters;
    if (!sems.length) { el.innerHTML = '<div class="empty-state"><div class="empty-icon">📖</div><div class="empty-title">No active semesters</div><div class="empty-desc">Create a semester first</div></div>'; return; }

    let html = '';
    for (const sem of sems) {
      const subjects = await API.get(`/api/semesters/${sem.id}/subjects`);
      html += `<div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--text2);margin-bottom:10px;padding:8px 12px;background:var(--bg3);border-radius:8px">🎓 ${escHtml(sem.name)}</div>`;
      if (!subjects.length) {
        html += `<div style="color:var(--text3);font-size:13px;padding:10px 12px">No subjects yet. <button class="btn btn-ghost btn-xs" onclick="document.getElementById('subjectSemSelect').value='${sem.id}';openModal('subjectModal')">+ Add one</button></div>`;
      } else {
        for (const sub of subjects) {
          const cos = await API.get(`/api/subjects/${sub.id}/course-outcomes`);
          html += `<div class="accordion" id="subAccordion-${sub.id}">
            <div class="accordion-header" onclick="toggleAccordion('subAccordion-${sub.id}')">
              <div class="accordion-title">
                <span class="badge badge-blue">${escHtml(sub.code)}</span>
                ${escHtml(sub.name)}
                ${sub.credits ? `<span style="font-size:12px;color:var(--text3)">${sub.credits}cr</span>` : ''}
                <span class="badge badge-purple">${cos.length} COs</span>
              </div>
              <div style="display:flex;gap:6px;align-items:center">
                <button class="btn btn-ghost btn-xs" onclick="event.stopPropagation();openAddCO(${sub.id})">+ CO</button>
                <button class="btn btn-ghost btn-xs" onclick="event.stopPropagation();openEditSubject(${sub.id},'${escHtml(sub.code)}','${escHtml(sub.name)}',${sub.credits||'null'})">✏️</button>
                <button class="btn btn-danger btn-xs" onclick="event.stopPropagation();deleteSubject(${sub.id},'${escHtml(sub.code)}')">✕</button>
                <span class="accordion-arrow">▾</span>
              </div>
            </div>
            <div class="accordion-body">
              ${cos.length ? cos.map(co => `
                <div class="sem-tree-co">
                  <span class="badge badge-purple" style="font-size:11px">${co.co_code}</span>
                  <span>${escHtml(co.title.substring(0,100))}${co.title.length>100?'...':''}</span>
                  <button class="btn btn-danger btn-xs" style="margin-left:auto" onclick="deleteCO(${co.id},'${co.co_code}')">✕</button>
                </div>`).join('') : '<div style="color:var(--text3);font-size:13px">No course outcomes yet.</div>'}
            </div>
          </div>`;
        }
      }
      html += `<button class="btn btn-ghost btn-sm" style="margin-top:8px" onclick="document.getElementById('subjectSemSelect').value='${sem.id}';openModal('subjectModal')">+ Add Subject to ${escHtml(sem.name)}</button>`;
      html += '</div>';
    }
    el.innerHTML = html;
  } catch(e) { el.innerHTML = `<div style="color:var(--red);padding:20px">${e.message}</div>`; }
}

function toggleAccordion(id) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
}

async function createSubject() {
  const sem_id = document.getElementById('subjectSemSelect').value;
  const code = document.getElementById('subjectCode').value.trim();
  const name = document.getElementById('subjectName').value.trim();
  const credits = document.getElementById('subjectCredits').value;
  if (!sem_id || !code || !name) { toast('Fill all required fields', 'warning'); return; }
  try {
    await API.post('/api/subjects', { semester_id: parseInt(sem_id), code, name, credits: credits ? parseInt(credits) : null });
    closeModal('subjectModal');
    ['subjectCode','subjectName','subjectCredits'].forEach(id => document.getElementById(id).value = '');
    toast('Subject added!', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteSubject(id, code) {
  if (!confirm(`Delete subject ${code}? OK = hard delete`)) return;
  try {
    await API.delete(`/api/subjects/${id}?mode=hard`);
    toast('Subject deleted', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

function openAddCO(subjectId) {
  document.getElementById('coSubjectId').value = subjectId;
  document.getElementById('coCode').value = '';
  document.getElementById('coTitle').value = '';
  document.getElementById('coDesc').value = '';
  openModal('coModal');
}

async function createCO() {
  const subject_id = parseInt(document.getElementById('coSubjectId').value);
  const co_code = document.getElementById('coCode').value.trim();
  const title = document.getElementById('coTitle').value.trim();
  const description = document.getElementById('coDesc').value.trim();
  if (!co_code || !title) { toast('Fill CO code and title', 'warning'); return; }
  try {
    await API.post('/api/course-outcomes', { subject_id, co_code, title, description });
    closeModal('coModal');
    toast('CO added!', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteCO(id, code) {
  if (!confirm(`Delete CO ${code}?`)) return;
  try {
    await API.delete(`/api/course-outcomes/${id}?mode=soft`);
    toast('CO hidden', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ── Edit Semester ──────────────────────────────────────
function openEditSemester(id, name, start, end) {
  document.getElementById('editSemId').value = id;
  document.getElementById('editSemName').value = name;
  document.getElementById('editSemStart').value = start ? start.substring(0,10) : '';
  document.getElementById('editSemEnd').value = end ? end.substring(0,10) : '';
  openModal('editSemModal');
}
async function saveSemesterEdit() {
  const id = document.getElementById('editSemId').value;
  const name = document.getElementById('editSemName').value.trim();
  const start = document.getElementById('editSemStart').value;
  const end = document.getElementById('editSemEnd').value;
  if (!name || !start || !end) { toast('Fill all fields', 'warning'); return; }
  try {
    await API.patch(`/api/semesters/${id}`, {name, start_date: start, end_date: end});
    closeModal('editSemModal');
    toast('Semester updated!', 'success');
    loadSemesters();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ── Edit Subject ───────────────────────────────────────
function openEditSubject(id, code, name, credits) {
  document.getElementById('editSubjectId').value = id;
  document.getElementById('editSubjectCode').value = code;
  document.getElementById('editSubjectName').value = name;
  document.getElementById('editSubjectCredits').value = credits || '';
  openModal('editSubjectModal');
}
async function saveSubjectEdit() {
  const id = document.getElementById('editSubjectId').value;
  const code = document.getElementById('editSubjectCode').value.trim();
  const name = document.getElementById('editSubjectName').value.trim();
  const credits = document.getElementById('editSubjectCredits').value;
  if (!code || !name) { toast('Code and name are required', 'warning'); return; }
  try {
    await API.patch(`/api/subjects/${id}`, {code, name, credits: credits ? parseInt(credits) : null});
    closeModal('editSubjectModal');
    toast('Subject updated!', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ── Edit CO ────────────────────────────────────────────
function openEditCO(id, code, title, description) {
  document.getElementById('editCOId').value = id;
  document.getElementById('editCOCode').value = code;
  document.getElementById('editCOTitle').value = title;
  document.getElementById('editCODesc').value = description || '';
  openModal('editCOModal');
}
async function saveCOEdit() {
  const id = document.getElementById('editCOId').value;
  const co_code = document.getElementById('editCOCode').value.trim();
  const title = document.getElementById('editCOTitle').value.trim();
  const description = document.getElementById('editCODesc').value.trim();
  if (!co_code || !title) { toast('CO code and title required', 'warning'); return; }
  try {
    await API.patch(`/api/course-outcomes/${id}`, {co_code, title, description});
    closeModal('editCOModal');
    toast('Course outcome updated!', 'success');
    loadSubjectsList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ── Edit Resource ──────────────────────────────────────
function openEditResource(id, title) {
  document.getElementById('editResourceId').value = id;
  document.getElementById('editResourceTitle').value = title;
  openModal('editResourceModal');
}
async function saveResourceEdit() {
  const id = document.getElementById('editResourceId').value;
  const title = document.getElementById('editResourceTitle').value.trim();
  if (!title) { toast('Title required', 'warning'); return; }
  try {
    await API.patch(`/api/resources/${id}`, {title});
    closeModal('editResourceModal');
    toast('Resource renamed!', 'success');
    loadResourcesList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}


// ═══════════════════════════════════════════════════════════
// RESOURCES
// ═══════════════════════════════════════════════════════════
async function loadResourcesPage() {
  try {
    const subs = await API.get('/api/subjects/all');
    const sel = document.getElementById('uploadSubjectSelect');
    sel.innerHTML = '<option value="">-- Select a subject first --</option>';
    subs.forEach(s => sel.innerHTML += `<option value="${s.id}">${s.code} — ${s.name}</option>`);
    loadResourcesList();
  } catch(e) {}
}

async function loadResourcesList() {
  const sel = document.getElementById('uploadSubjectSelect');
  const subjectId = sel.value;
  if (!subjectId) {
    document.getElementById('resourcesList').innerHTML = '<div style="color:var(--text2);font-size:14px">Select a subject to view its resources</div>';
    return;
  }
  try {
    const res = await API.get(`/api/resources/subject/${subjectId}`);
    const el = document.getElementById('resourcesList');
    if (!res.length) { el.innerHTML = '<div style="color:var(--text2);font-size:14px;padding:16px 0">No resources uploaded yet</div>'; return; }
    el.innerHTML = `<div class="section-title" style="margin-bottom:12px">Uploaded Resources</div>` + res.map(r => `
      <div class="card card-sm" style="margin-bottom:8px;display:flex;align-items:center;gap:12px">
        <span style="font-size:24px">📄</span>
        <div style="flex:1">
          <div style="font-weight:500">${escHtml(r.title)}</div>
          <div style="font-size:12px;color:var(--text2)">${fmtDate(r.uploaded_at)} · ${r.raw_text ? Math.round(r.raw_text.length/1000)+'K chars extracted' : 'No text'}</div>
        </div>
        <button class="btn btn-ghost btn-xs" onclick="openEditResource(${r.id},'${escHtml(r.title)}')">✏️</button>
        <button class="btn btn-danger btn-xs" onclick="deleteResource(${r.id})">✕</button>
      </div>`).join('');
  } catch(e) {}
}

document.getElementById('uploadSubjectSelect').addEventListener('change', loadResourcesList);

function handleFileUpload(input) {
  if (input.files[0]) processUpload(input.files[0]);
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.pdf')) processUpload(file);
  else toast('Please drop a PDF file', 'warning');
}

async function processUpload(file) {
  const subjectId = document.getElementById('uploadSubjectSelect').value;
  if (!subjectId) { toast('Please select a subject first', 'warning'); return; }
  document.getElementById('uploadProgress').style.display = 'block';
  document.getElementById('uploadZone').style.display = 'none';
  try {
    const formData = new FormData();
    formData.append('file', file);
    const r = await fetch(`/api/resources/upload?subject_id=${subjectId}`, { method: 'POST', body: formData });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    const ai = data.ai_extraction;
    toast(`📄 Extracted! ${ai.course_outcomes?.length||0} COs found via ${ai.method||'pattern'}`, 'success');
    if (ai.course_outcomes?.length) showCOReview(ai.course_outcomes, parseInt(subjectId));
    loadResourcesList();
  } catch(e) { toast('Upload error: ' + e.message, 'error'); }
  document.getElementById('uploadProgress').style.display = 'none';
  document.getElementById('uploadZone').style.display = 'block';
}

function showCOReview(cos, subjectId) {
  S.coReviewSubjectId = subjectId;
  document.getElementById('coReviewMeta').textContent = `Found ${cos.length} course outcomes. Review and edit before saving.`;
  document.getElementById('coReviewList').innerHTML = cos.map((co, i) => `
    <div style="display:grid;grid-template-columns:80px 1fr auto;gap:8px;margin-bottom:8px;align-items:start">
      <input class="input" value="${escHtml(co.co_code)}" id="reviewCoCode-${i}" placeholder="CO1">
      <input class="input" value="${escHtml(co.title)}" id="reviewCoTitle-${i}" placeholder="Title">
      <button class="btn btn-danger btn-icon" onclick="this.parentElement.remove()">✕</button>
    </div>`).join('');
  openModal('coReviewModal');
}

function addCOReviewItem() {
  const list = document.getElementById('coReviewList');
  const n = list.querySelectorAll('[id^="reviewCoCode-"]').length;
  const div = document.createElement('div');
  div.style.cssText = 'display:grid;grid-template-columns:80px 1fr auto;gap:8px;margin-bottom:8px;align-items:start';
  div.innerHTML = `<input class="input" id="reviewCoCode-${n}" placeholder="CO${n+1}"><input class="input" id="reviewCoTitle-${n}" placeholder="Title"><button class="btn btn-danger btn-icon" onclick="this.parentElement.remove()">✕</button>`;
  list.appendChild(div);
}

async function saveCOsFromReview() {
  const list = document.getElementById('coReviewList');
  const rows = list.querySelectorAll('[id^="reviewCoCode-"]');
  const cos = [];
  rows.forEach((el, i) => {
    const code = el.value.trim();
    const titleEl = list.querySelector(`#reviewCoTitle-${i}`);
    if (!titleEl) return;
    const title = titleEl.value.trim();
    if (code && title) cos.push({ subject_id: S.coReviewSubjectId, co_code: code, title, description: '' });
  });
  if (!cos.length) { toast('No COs to save', 'warning'); return; }
  try {
    await API.post('/api/course-outcomes/bulk', cos);
    closeModal('coReviewModal');
    toast(`✅ Saved ${cos.length} course outcomes!`, 'success');
    if (S.currentPage === 'subjects') loadSubjectsList();
  } catch(e) { toast('Error saving COs: ' + e.message, 'error'); }
}

async function deleteResource(id) {
  if (!confirm('Delete this resource?')) return;
  try {
    await API.delete(`/api/resources/${id}`);
    toast('Resource deleted', 'success');
    loadResourcesList();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════════════════════════
// PROJECTS
// ═══════════════════════════════════════════════════════════
async function loadProjects(status = 'active', btn) {
  if (btn) {
    btn.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
  }
  const el = document.getElementById('projectsList');
  el.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const projs = await API.get(`/api/projects?status=${status}`);
    if (!projs.length) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">🚀</div><div class="empty-title">No ${status} projects</div><div class="empty-desc">Create a project to track your work</div></div>`;
      return;
    }
    el.innerHTML = projs.map(p => `
      <div class="card" style="margin-bottom:14px" id="proj-${p.id}">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:12px">
          <div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:16px">${escHtml(p.name)}</div>
            ${p.description ? `<div style="font-size:13px;color:var(--text2);margin-top:4px">${escHtml(p.description)}</div>` : ''}
          </div>
          <div style="display:flex;gap:6px">
            <span class="badge badge-${p.priority==='high'?'red':p.priority==='medium'?'yellow':'green'}">${p.priority}</span>
            ${p.deadline ? `<span class="badge badge-blue">📅 ${fmtDate(p.deadline)}</span>` : ''}
            <button class="btn btn-ghost btn-xs" onclick="toggleProjStatus(${p.id},'${p.status}')">${p.status==='active'?'✓ Done':'↩ Reopen'}</button>
            <button class="btn btn-danger btn-xs" onclick="deleteProject(${p.id})">✕</button>
          </div>
        </div>
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:13px;color:var(--text2)">
            <span>${p.done_count}/${p.task_count} tasks</span><span style="font-weight:700;color:var(--text)">${p.progress}%</span>
          </div>
          <div class="progress-bar" style="height:8px"><div class="progress-fill ${p.progress>=80?'progress-green':p.progress>=50?'progress-yellow':'progress-purple'}" style="width:${p.progress}%"></div></div>
        </div>
        <div id="projTasks-${p.id}">
          ${p.tasks.map(t => `
            <div class="task-item ${t.done?'done':''}" id="ptask-${t.id}">
              <div class="task-cb ${t.done?'checked':''}" onclick="toggleProjTask(${t.id},${p.id},${!t.done})"></div>
              <div class="task-text">${escHtml(t.text)}</div>
              <div class="task-actions"><button class="btn btn-xs btn-danger" onclick="deleteProjTask(${t.id},${p.id})">✕</button></div>
            </div>`).join('')}
        </div>
        <div style="display:flex;gap:8px;margin-top:10px">
          <input class="input" id="newProjTask-${p.id}" placeholder="Add task..." style="flex:1" onkeypress="if(event.key==='Enter')addProjTask(${p.id})">
          <button class="btn btn-ghost btn-sm" onclick="addProjTask(${p.id})">+ Add</button>
        </div>
      </div>`).join('');
  } catch(e) { el.innerHTML = `<div style="color:var(--red)">${e.message}</div>`; }
}

async function createProject() {
  const name = document.getElementById('projName').value.trim();
  if (!name) { toast('Enter project name', 'warning'); return; }
  const desc = document.getElementById('projDesc').value.trim();
  const priority = document.getElementById('projPriority').value;
  const deadline = document.getElementById('projDeadline').value || null;
  try {
    await API.post('/api/projects', { name, description: desc, priority, deadline });
    closeModal('projectModal');
    ['projName','projDesc','projDeadline'].forEach(id => document.getElementById(id).value = '');
    toast('Project created!', 'success');
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function addProjTask(projId) {
  const input = document.getElementById(`newProjTask-${projId}`);
  const text = input.value.trim();
  if (!text) return;
  try {
    await API.post(`/api/projects/${projId}/tasks`, { text });
    input.value = '';
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function toggleProjTask(taskId, projId, done) {
  try {
    await API.patch(`/api/projects/tasks/${taskId}`, { done });
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteProjTask(taskId, projId) {
  try {
    await API.delete(`/api/projects/tasks/${taskId}`);
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function toggleProjStatus(id, current) {
  const newStatus = current === 'active' ? 'done' : 'active';
  try {
    await API.patch(`/api/projects/${id}`, { status: newStatus });
    toast(`Project ${newStatus === 'done' ? 'completed' : 'reopened'}!`, 'success');
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteProject(id) {
  if (!confirm('Delete this project permanently?')) return;
  try {
    await API.delete(`/api/projects/${id}`);
    toast('Project deleted', 'success');
    loadProjects();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════════════════════════
// NOTES
// ═══════════════════════════════════════════════════════════
async function loadNotes(query = '') {
  const el = document.getElementById('notesList');
  el.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const notes = await API.get(`/api/notes${query ? '?q=' + encodeURIComponent(query) : ''}`);
    if (!notes.length) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">📝</div><div class="empty-title">No notes yet</div><div class="empty-desc">Capture your thoughts, ideas, and important info</div></div>`;
      return;
    }
    el.innerHTML = `<div class="grid grid-3">${notes.map(n => `
      <div class="card" style="cursor:pointer" onclick="editNote(${n.id},'${escHtml(n.title)}','${escHtml(n.content||'')}','${escHtml(n.tags||'')}')">
        <div style="font-weight:600;margin-bottom:8px">${escHtml(n.title)}</div>
        ${n.content ? `<div style="font-size:13px;color:var(--text2);margin-bottom:10px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">${escHtml(n.content)}</div>` : ''}
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>${(n.tags||'').split(',').filter(Boolean).map(t => `<span class="tag">${t.trim()}</span>`).join('')}</div>
          <div style="display:flex;gap:4px">
            <span style="font-size:11px;color:var(--text3)">${fmtDate(n.updated_at)}</span>
            <button class="btn btn-danger btn-xs" onclick="event.stopPropagation();deleteNote(${n.id})">✕</button>
          </div>
        </div>
      </div>`).join('')}</div>`;
  } catch(e) { el.innerHTML = `<div style="color:var(--red)">${e.message}</div>`; }
}

let noteSearchTimer;
function searchNotes(q) {
  clearTimeout(noteSearchTimer);
  noteSearchTimer = setTimeout(() => loadNotes(q), 300);
}

function editNote(id, title, content, tags) {
  S.editingNoteId = id;
  document.getElementById('noteModalTitle').textContent = 'Edit Note';
  document.getElementById('editNoteId').value = id;
  document.getElementById('noteTitle').value = title;
  document.getElementById('noteContent').value = content;
  document.getElementById('noteTags').value = tags;
  openModal('noteModal');
}

async function saveNote() {
  const title = document.getElementById('noteTitle').value.trim();
  if (!title) { toast('Enter note title', 'warning'); return; }
  const content = document.getElementById('noteContent').value;
  const tags = document.getElementById('noteTags').value;
  const editId = document.getElementById('editNoteId').value;
  try {
    if (editId) {
      await API.patch(`/api/notes/${editId}`, { title, content, tags });
    } else {
      await API.post('/api/notes', { title, content, tags });
    }
    closeModal('noteModal');
    document.getElementById('editNoteId').value = '';
    ['noteTitle','noteContent','noteTags'].forEach(id => document.getElementById(id).value = '');
    document.getElementById('noteModalTitle').textContent = 'New Note';
    toast('Note saved!', 'success');
    loadNotes();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

async function deleteNote(id) {
  if (!confirm('Delete this note?')) return;
  try {
    await API.delete(`/api/notes/${id}`);
    toast('Note deleted', 'success');
    loadNotes();
  } catch(e) { toast('Error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════════════════════════
// AI CHAT  (Groq AI)
// ═══════════════════════════════════════════════════════════
async function loadChatHistory() {
  try {
    const msgs = await API.get('/api/ai/chat/history?limit=20');
    if (msgs.length) {
      const el = document.getElementById('chatMessages');
      el.innerHTML = '';
      msgs.forEach(m => appendChatMsg(m.role, m.content, false));
      el.scrollTop = el.scrollHeight;
    }
  } catch(e) {}
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  appendChatMsg('user', msg);
  const typing = showTyping();
  try {
    const data = await API.post('/api/ai/chat', { message: msg, context: 'general' });
    typing.remove();
    appendChatMsg('assistant', data.response);
  } catch(e) {
    typing.remove();
    appendChatMsg('assistant', '❌ Error: ' + e.message);
  }
}

function sendQuick(msg) {
  document.getElementById('chatInput').value = msg;
  sendChat();
}

function appendChatMsg(role, content, scroll = true) {
  const el = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<div class="chat-bubble">${escHtml(content).replace(/\n/g,'<br>')}</div>`;
  el.appendChild(div);
  if (scroll) el.scrollTop = el.scrollHeight;
}

function showTyping() {
  const el = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  div.innerHTML = `<div class="chat-typing"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div;
}

async function clearChat() {
  if (!confirm('Clear all chat history?')) return;
  await API.delete('/api/ai/chat/history');
  document.getElementById('chatMessages').innerHTML = `<div class="chat-msg assistant"><div class="chat-bubble">Chat cleared! Ask me anything 🤖</div></div>`;
}

// ═══════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════
function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function fmtDate(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  } catch { return dateStr; }
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
(async function init() {
  loadDashboard();

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const map = { 'd': 'dashboard', 't': 'today', 'p': 'progress', 's': 'semesters', 'j': 'projects', 'n': 'notes', 'a': 'ai' };
    if (map[e.key]) navigate(map[e.key]);
  });
})();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)

@app.get("/health")
def health():
    return {"status": "healthy", "db": "connected", "groq": bool(_groq_client), "model": GROQ_MODEL}

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"""
╔══════════════════════════════════════════════════════╗
║           DayMark Student OS v2.0 Starting           ║
╠══════════════════════════════════════════════════════╣
║  URL:     http://localhost:{port:<26}║
║  DB:      {DATABASE_URL[:45]:<46}║
║  Groq:    {'✅ Active (' + GROQ_MODEL + ')' if _groq_client else '⚠️  Set GROQ_API_KEY in .env':<46}║
╠══════════════════════════════════════════════════════╣
║  Key:     https://console.groq.com                   ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("daymark_app:app", host="0.0.0.0", port=port, reload=True)