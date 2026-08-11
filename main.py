import os
import json
import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition
import pandas as pd

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Date,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# ============================================================
# 1. PATHS & DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(BASE_DIR / "data")
    )
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "eduvision.db"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DB_PATH}"
)

# Render/Postgres compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )

connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# ============================================================
# 2. DATABASE MODELS
# ============================================================

class User(Base):
    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True
    )

    username = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    password = Column(
        String,
        nullable=False
    )

    role = Column(
        String,
        nullable=False
    )

    phone_number = Column(
        String,
        nullable=True
    )


class Student(Base):
    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True
    )

    name = Column(
        String,
        index=True,
        nullable=False
    )

    roll_number = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    class_section = Column(
        String,
        nullable=False
    )

    parent_id = Column(
        Integer,
        nullable=True
    )

    # Multiple face samples can be stored here.
    face_encodings = Column(
        Text,
        default="[]"
    )


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(
        Integer,
        primary_key=True
    )

    student_id = Column(
        Integer,
        nullable=False
    )

    date = Column(
        Date,
        nullable=False
    )

    timestamp = Column(
        DateTime,
        nullable=False
    )

    session_type = Column(
        String,
        nullable=False
    )

    status = Column(
        String,
        nullable=False
    )

    marked_by = Column(
        String,
        nullable=False
    )


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(
        Integer,
        primary_key=True
    )

    parent_id = Column(
        Integer,
        nullable=True
    )

    student_id = Column(
        Integer,
        nullable=True
    )

    message = Column(
        String,
        nullable=False
    )

    timestamp = Column(
        DateTime,
        default=datetime.now
    )

    is_read = Column(
        Boolean,
        default=False
    )


# Create tables
Base.metadata.create_all(bind=engine)


# ============================================================
# 3. DATABASE SESSION
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# 4. DEMO DATA
# ============================================================

def init_demo_data():

    db = SessionLocal()

    try:

        if db.query(User).count() == 0:

            admin = User(
                username="admin",
                password="password",
                role="Admin",
                phone_number=""
            )

            teacher = User(
                username="teacher",
                password="password",
                role="Teacher",
                phone_number=""
            )

            parent = User(
                username="parent",
                password="password",
                role="Parent",
                phone_number=""
            )

            db.add_all([
                admin,
                teacher,
                parent
            ])

            db.commit()

            db.refresh(parent)

            demo_student = Student(
                name="Rahul Sharma",
                roll_number="CS-101",
                class_section="10-A",
                parent_id=parent.id,
                face_encodings="[]"
            )

            db.add(demo_student)

            db.commit()

    finally:

        db.close()


init_demo_data()


# ============================================================
# 5. FASTAPI APP
# ============================================================

app = FastAPI(
    title="EduvisionAI",
    description="Smart Multi-Face AI Attendance System",
    version="2.0.0"
)


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# 7. REQUEST MODELS
# ============================================================

class LoginData(BaseModel):

    username: str

    password: str


class EnrollRequest(BaseModel):

    student_id: int

    image_base64: str


class FrameRequest(BaseModel):

    image_base64: str


class ManualAttendance(BaseModel):

    student_id: int

    session_type: str


# ============================================================
# 8. IMAGE DECODER
# ============================================================

def decode_image(data_url: str):

    """
    Converts browser base64 image
    into OpenCV image.
    """

    if "," in data_url:

        raw = data_url.split(",", 1)[1]

    else:

        raw = data_url

    try:

        image_bytes = base64.b64decode(raw)

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    return image


# ============================================================
# 9. LOAD KNOWN FACE ENCODINGS
# ============================================================

def get_known_faces(db: Session):

    known_encodings = []

    student_ids = []

    student_names = []

    students = db.query(Student).all()

    for student in students:

        try:

            saved_encodings = json.loads(
                student.face_encodings or "[]"
            )

        except Exception:

            continue

        for encoding in saved_encodings:

            vector = np.asarray(
                encoding,
                dtype=np.float64
            )

            if vector.shape != (128,):

                continue

            known_encodings.append(vector)

            student_ids.append(
                student.id
            )

            student_names.append(
                student.name
            )

    return (
        known_encodings,
        student_ids,
        student_names
    )


# ============================================================
# 10. ATTENDANCE ENGINE
# ============================================================

def mark_attendance(
    student_id: int,
    db: Session,
    marked_by="CCTV"
):

    now = datetime.now()

    # Before 12 PM = Check-In
    # 12 PM onwards = Check-Out

    session_type = (
        "Check-In"
        if now.hour < 12
        else "Check-Out"
    )

    # Prevent duplicate attendance
    existing = db.query(
        Attendance
    ).filter(
        Attendance.student_id == student_id,
        Attendance.date == now.date(),
        Attendance.session_type == session_type
    ).first()

    if existing:

        return None

    student = db.query(
        Student
    ).filter(
        Student.id == student_id
    ).first()

    if not student:

        return None

    attendance = Attendance(
        student_id=student_id,
        date=now.date(),
        timestamp=now,
        session_type=session_type,
        status=(
            "Present"
            if marked_by == "CCTV"
            else "Manual Override"
        ),
        marked_by=marked_by
    )

    db.add(attendance)

    # Parent notification record
    if student.parent_id:

        notification = Notification(
            parent_id=student.parent_id,
            student_id=student.id,
            message=(
                f"EduvisionAI: "
                f"{student.name} "
                f"{session_type.lower()} "
                f"recorded at "
                f"{now.strftime('%I:%M %p')}."
            ),
            timestamp=now
        )

        db.add(notification)

    db.commit()

    return {
        "student_id": student.id,
        "name": student.name,
        "session": session_type,
        "time": now.strftime("%I:%M %p")
    }


# ============================================================
# 11. HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        BASE_DIR / "index.html"
    )


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "EduvisionAI",
        "face_recognition": True,
        "multi_face": True
    }


# ============================================================
# 13. LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(
    data: LoginData,
    db: Session = Depends(get_db)
):

    user = db.query(
        User
    ).filter(
        User.username == data.username,
        User.password == data.password
    ).first()

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    return {
        "id": user.id,
        "username": user.username,
        "role": user.role
    }


# ============================================================
# 14. FACE ENROLLMENT
# ============================================================

@app.post("/api/admin/enroll")
def enroll_face(
    data: EnrollRequest,
    db: Session = Depends(get_db)
):

    student = db.query(
        Student
    ).filter(
        Student.id == data.student_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    image = decode_image(
        data.image_base64
    )

    # BGR -> RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Detect all faces
    face_locations = (
        face_recognition.face_locations(
            rgb_image,
            model="hog"
        )
    )

    # Enrollment must contain exactly one face
    if len(face_locations) == 0:

        raise HTTPException(
            status_code=400,
            detail="No face detected."
        )

    if len(face_locations) > 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "Multiple faces detected. "
                "Only one student should be "
                "visible during enrollment."
            )
        )

    encodings = (
        face_recognition.face_encodings(
            rgb_image,
            face_locations
        )
    )

    if not encodings:

        raise HTTPException(
            status_code=400,
            detail="Could not generate face encoding."
        )

    new_encoding = encodings[0].tolist()

    try:

        current_encodings = json.loads(
            student.face_encodings or "[]"
        )

    except Exception:

        current_encodings = []

    # Keep maximum 3 samples
    current_encodings.append(
        new_encoding
    )

    current_encodings = current_encodings[-3:]

    student.face_encodings = json.dumps(
        current_encodings
    )

    db.commit()

    return {
        "success": True,
        "student": student.name,
        "samples": len(current_encodings),
        "message": (
            f"Face enrolled successfully "
            f"for {student.name}"
        )
    }


# ============================================================
# 15. MULTI-FACE RECOGNITION
# ============================================================

@app.post("/api/camera/frame")
def process_frame(
    data: FrameRequest,
    db: Session = Depends(get_db)
):

    image = decode_image(
        data.image_base64
    )

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Detect ALL faces in current frame
    # --------------------------------------------------------

    face_locations = (
        face_recognition.face_locations(
            rgb_image,
            model="hog"
        )
    )

    face_encodings = (
        face_recognition.face_encodings(
            rgb_image,
            face_locations
        )
    )

    # --------------------------------------------------------
    # Load enrolled students
    # --------------------------------------------------------

    (
        known_encodings,
        student_ids,
        student_names
    ) = get_known_faces(db)

    detections = []

    newly_marked = []

    # --------------------------------------------------------
    # Process every face independently
    # --------------------------------------------------------

    for location, face_encoding in zip(
        face_locations,
        face_encodings
    ):

        top, right, bottom, left = location

        name = "Unknown"

        student_id = None

        distance = None

        # ----------------------------------------------------
        # Match against ALL enrolled faces
        # ----------------------------------------------------

        if known_encodings:

            distances = (
                face_recognition.face_distance(
                    known_encodings,
                    face_encoding
                )
            )

            best_index = int(
                np.argmin(distances)
            )

            best_distance = float(
                distances[best_index]
            )

            distance = best_distance

            # Strict threshold
            MATCH_THRESHOLD = 0.48

            if best_distance <= MATCH_THRESHOLD:

                student_id = (
                    student_ids[best_index]
                )

                name = (
                    student_names[best_index]
                )

                # ------------------------------------------------
                # Attendance
                # ------------------------------------------------

                attendance = mark_attendance(
                    student_id,
                    db,
                    marked_by="CCTV"
                )

                if attendance:

                    newly_marked.append(
                        attendance
                    )

        detections.append({

            "name": name,

            "student_id": student_id,

            "distance": (
                round(distance, 4)
                if distance is not None
                else None
            ),

            "box": {

                "top": top,

                "right": right,

                "bottom": bottom,

                "left": left
            }
        })

    return {

        "faces": detections,

        "attendance": newly_marked,

        "face_count": len(detections)
    }


# ============================================================
# 16. RECENT ATTENDANCE
# ============================================================

@app.get("/api/attendance/recent")
def recent_attendance(
    db: Session = Depends(get_db)
):

    today = datetime.now().date()

    rows = (
        db.query(
            Attendance,
            Student
        )
        .join(
            Student,
            Attendance.student_id == Student.id
        )
        .filter(
            Attendance.date == today
        )
        .order_by(
            Attendance.timestamp.desc()
        )
        .limit(30)
        .all()
    )

    return [

        {
            "name": student.name,

            "session": attendance.session_type,

            "time": attendance.timestamp.strftime(
                "%I:%M %p"
            )
        }

        for attendance, student in rows
    ]


# ============================================================
# 17. TEACHER STUDENT ROSTER
# ============================================================

@app.get("/api/teacher/students")
def teacher_students(
    db: Session = Depends(get_db)
):

    students = db.query(
        Student
    ).all()

    today = datetime.now().date()

    result = []

    for student in students:

        logs = db.query(
            Attendance
        ).filter(
            Attendance.student_id == student.id,
            Attendance.date == today
        ).all()

        has_check_in = any(
            log.session_type == "Check-In"
            for log in logs
        )

        has_check_out = any(
            log.session_type == "Check-Out"
            for log in logs
        )

        result.append({

            "id": student.id,

            "name": student.name,

            "roll": student.roll_number,

            "class": student.class_section,

            "has_in": has_check_in,

            "has_out": has_check_out
        })

    return result


# ============================================================
# 18. MANUAL ATTENDANCE
# ============================================================

@app.post("/api/teacher/mark")
def teacher_mark(
    data: ManualAttendance,
    db: Session = Depends(get_db)
):

    if data.session_type not in {
        "Check-In",
        "Check-Out"
    }:

        raise HTTPException(
            status_code=400,
            detail="Invalid session type."
        )

    result = mark_attendance(
        data.student_id,
        db,
        marked_by="Teacher"
    )

    if result:

        return result

    return {
        "success": True,
        "message": "Attendance already marked."
    }


# ============================================================
# 19. PARENT DASHBOARD
# ============================================================

@app.get("/api/parent/{parent_id}/dashboard")
def parent_dashboard(
    parent_id: int,
    db: Session = Depends(get_db)
):

    student = db.query(
        Student
    ).filter(
        Student.parent_id == parent_id
    ).first()

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Child record not found."
        )

    today = datetime.now().date()

    logs = db.query(
        Attendance
    ).filter(
        Attendance.student_id == student.id,
        Attendance.date == today
    ).all()

    has_check_in = any(
        log.session_type == "Check-In"
        for log in logs
    )

    has_check_out = any(
        log.session_type == "Check-Out"
        for log in logs
    )

    if not has_check_in:

        status = "🏡 Not Arrived Yet"

    elif has_check_in and not has_check_out:

        status = "🎒 In Class Now"

    else:

        status = "🚌 Dismissed"

    notifications = (
        db.query(Notification)
        .filter(
            Notification.parent_id == parent_id
        )
        .order_by(
            Notification.timestamp.desc()
        )
        .limit(20)
        .all()
    )

    timeline = [

        {
            "msg": notification.message,

            "time": notification.timestamp.strftime(
                "%Y-%m-%d %I:%M %p"
            )
        }

        for notification in notifications
    ]

    return {

        "student_name": student.name,

        "status": status,

        "timeline": timeline
    }


# ============================================================
# 20. EXCEL EXPORT
# ============================================================

@app.get("/api/export")
def export_excel(
    db: Session = Depends(get_db)
):

    rows = (
        db.query(
            Attendance,
            Student
        )
        .join(
            Student,
            Attendance.student_id == Student.id
        )
        .order_by(
            Attendance.timestamp.desc()
        )
        .all()
    )

    data = []

    for attendance, student in rows:

        data.append({

            "Date":
                attendance.date.strftime(
                    "%Y-%m-%d"
                ),

            "Roll Number":
                student.roll_number,

            "Student Name":
                student.name,

            "Class":
                student.class_section,

            "Session":
                attendance.session_type,

            "Status":
                attendance.status,

            "Time":
                attendance.timestamp.strftime(
                    "%I:%M %p"
                ),

            "Marked By":
                attendance.marked_by
        })

    report_path = (
        DATA_DIR /
        "attendance_report.xlsx"
    )

    dataframe = pd.DataFrame(data)

    dataframe.to_excel(
        report_path,
        index=False
    )

    return FileResponse(
        report_path,
        filename="EduvisionAI_Attendance_Report.xlsx",
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# 21. OLD CCTV ENDPOINT
# ============================================================

@app.get("/api/camera/stream")
def old_camera_stream():

    raise HTTPException(

        status_code=410,

        detail=(
            "Server webcam streaming has been removed. "
            "Use browser camera frames through "
            "/api/camera/frame."
        )
    )


# ============================================================
# 22. LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
