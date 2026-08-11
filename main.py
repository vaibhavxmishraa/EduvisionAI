import os
import json
import base64
import asyncio
from datetime import datetime, date, time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Date, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Check computer vision dependencies
HAS_CV = False
try:
    import cv2
    import numpy as np
    import face_recognition
    HAS_CV = True
    print("✅ Hardware & AI Libraries (OpenCV, Face_Recognition) loaded successfully.")
except ImportError:
    print("⚠️ WARNING: cv2 or face_recognition not found. Running in SIMULATION MODE.")

# Fallback for Pandas (Excel Export)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️ WARNING: pandas not found. Excel export will return JSON fallback.")

SQLALCHEMY_DATABASE_URL = "sqlite:///./eduvision.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String) # Admin, Teacher, Parent
    phone_number = Column(String)

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    roll_number = Column(String, unique=True, index=True)
    class_section = Column(String)
    parent_id = Column(Integer)
    face_encodings = Column(Text) # JSON stringified list of vectors

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer)
    date = Column(Date)
    timestamp = Column(DateTime)
    session_type = Column(String) # 'Check-In' or 'Check-Out'
    status = Column(String) # 'Present', 'Absent', 'Manual Override'
    marked_by = Column(String) # 'CCTV' or 'Teacher'

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer)
    student_id = Column(Integer)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    is_read = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_mock_data():
    db = SessionLocal()
    if db.query(User).count() == 0:
        print("🌱 Seeding initial mock data for EduvisionAI...")
        db.add_all([
            User(username="admin", password="password", role="Admin", phone_number="1234567890"),
            User(username="teacher", password="password", role="Teacher", phone_number="0987654321"),
            User(username="parent", password="password", role="Parent", phone_number="+15551234567")
        ])
        db.commit()
        
        parent_user = db.query(User).filter(User.username == "parent").first()
        db.add(Student(
            name="Rahul Sharma", 
            roll_number="CS-101", 
            class_section="10-A", 
            parent_id=parent_user.id,
            face_encodings="[]"
        ))
        db.commit()
        print("✅ Seed completed.")
    db.close()

init_mock_data()

app = FastAPI(title="EduvisionAI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def send_sms_alert(phone_number: str, message: str):
    try:
        from twilio.rest import Client
        TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "mock_sid")
        TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "mock_token")
        TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER", "+1234567890")
        
        if TWILIO_SID == "mock_sid":
            print(f"📱 [TWILIO MOCK SIMULATION] SMS to {phone_number}: {message}")
            return
            
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=phone_number
        )
        print(f"✅ Real SMS Sent: {msg.sid}")
    except Exception as e:
        print(f"⚠️ Twilio Error: {e}")

camera_active = False

def process_attendance(student_id: int, db: Session):
    now = datetime.now()
    current_date = now.date()
    
    # Morning Check-In (< 12:00) vs Evening Check-Out (>= 12:00)
    session_type = "Check-In" if now.hour < 12 else "Check-Out"
    
    existing = db.query(Attendance).filter(
        Attendance.student_id == student_id,
        Attendance.date == current_date,
        Attendance.session_type == session_type
    ).first()
    
    if existing:
        return None
        
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return None

    new_log = Attendance(
        student_id=student_id,
        date=current_date,
        timestamp=now,
        session_type=session_type,
        status="Present",
        marked_by="CCTV"
    )
    db.add(new_log)
    
    time_str = now.strftime("%I:%M %p")
    action_str = "arrived at school" if session_type == "Check-In" else "left school"
    msg = f"EduvisionAI Alert: Your child {student.name} has {action_str} at {time_str}."
    
    notif = Notification(
        parent_id=student.parent_id,
        student_id=student_id,
        message=msg,
        timestamp=now
    )
    db.add(notif)
    db.commit()
    
    parent = db.query(User).filter(User.id == student.parent_id).first()
    if parent and parent.phone_number:
        send_sms_alert(parent.phone_number, msg)
        
    return {"student": student.name, "session": session_type, "time": time_str}

def generate_cctv_frames():
    global camera_active
    camera_active = True
    
    if not HAS_CV:
        while camera_active:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            import time; time.sleep(1)
        return

    cap = cv2.VideoCapture(0)
    db = SessionLocal()
    
    known_face_encodings = []
    known_face_ids = []
    known_face_names = []
    students = db.query(Student).all()
    for s in students:
        encs = json.loads(s.face_encodings)
        if len(encs) > 0:
            known_face_encodings.append(np.array(encs[0]))
            known_face_ids.append(s.id)
            known_face_names.append(s.name)

    frame_count = 0
    face_locations = []
    face_names = []
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % 3 == 0 and len(known_face_encodings) > 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            detected_students_this_frame = []

            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.45)
                name = "Unknown"
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        student_id = known_face_ids[best_match_index]
                        name = known_face_names[best_match_index]
                        
                        res = process_attendance(student_id, db)
                        if res:
                            detected_students_this_frame.append(res['student'])
                
                face_names.append(name)

            if detected_students_this_frame:
                print(f"🟢 REAL-WORLD CLASS BATCH DETECTED ({len(detected_students_this_frame)} students): {', '.join(detected_students_this_frame)}")

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        frame_count += 1
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()
    db.close()

class LoginData(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username, User.password == data.password).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"id": user.id, "role": user.role, "username": user.username}

@app.get("/api/camera/stream")
def cctv_stream():
    return StreamingResponse(generate_cctv_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

class EnrollRequest(BaseModel):
    student_id: int
    image_base64: str

@app.post("/api/admin/enroll")
def enroll_face(data: EnrollRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == data.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
        
    if not HAS_CV:
        student.face_encodings = json.dumps([[0.1, 0.2, 0.3]])
        db.commit()
        return {"msg": "Simulation mode: Face vector saved successfully."}
        
    try:
        img_str = data.image_base64.split(",")[1] if "," in data.image_base64 else data.image_base64
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) == 0:
            raise HTTPException(status_code=400, detail="No face detected in capture.")
            
        current_encs = json.loads(student.face_encodings)
        current_encs.append(encodings[0].tolist())
        student.face_encodings = json.dumps(current_encs)
        db.commit()
        return {"msg": "Face successfully enrolled into vector database."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/attendance/recent")
def get_recent_attendance(db: Session = Depends(get_db)):
    today = datetime.now().date()
    logs = db.query(Attendance, Student).join(Student, Attendance.student_id == Student.id)\
             .filter(Attendance.date == today)\
             .order_by(Attendance.timestamp.desc()).limit(5).all()
             
    result = []
    for att, stu in logs:
        result.append({
            "name": stu.name,
            "session": att.session_type,
            "time": att.timestamp.strftime("%I:%M %p")
        })
    return result

@app.get("/api/parent/{parent_id}/dashboard")
def get_parent_dashboard(parent_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.parent_id == parent_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Child record not found")
        
    today = datetime.now().date()
    logs = db.query(Attendance).filter(Attendance.student_id == student.id, Attendance.date == today).all()
    
    status = "🏡 Not Arrived Yet"
    has_in = any(l.session_type == "Check-In" for l in logs)
    has_out = any(l.session_type == "Check-Out" for l in logs)
    
    if has_in and not has_out:
        status = "🎒 In Class Now"
    elif has_in and has_out:
        status = "🚌 Dismissed"
        
    notifications = db.query(Notification).filter(Notification.parent_id == parent_id)\
                      .order_by(Notification.timestamp.desc()).limit(10).all()
                      
    return {
        "student_name": student.name,
        "status": status,
        "timeline": [{"msg": n.message, "time": n.timestamp.strftime("%Y-%m-%d %I:%M %p")} for n in notifications]
    }

@app.get("/api/teacher/students")
def get_teacher_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    today = datetime.now().date()
    result = []
    for s in students:
        att = db.query(Attendance).filter(Attendance.student_id == s.id, Attendance.date == today).all()
        result.append({
            "id": s.id,
            "name": s.name,
            "roll": s.roll_number,
            "class": s.class_section,
            "has_in": any(l.session_type == "Check-In" for l in att),
            "has_out": any(l.session_type == "Check-Out" for l in att)
        })
    return result

class ManualAttendance(BaseModel):
    student_id: int
    session_type: str

@app.post("/api/teacher/mark")
def mark_manual(data: ManualAttendance, db: Session = Depends(get_db)):
    now = datetime.now()
    existing = db.query(Attendance).filter(
        Attendance.student_id == data.student_id,
        Attendance.date == now.date(),
        Attendance.session_type == data.session_type
    ).first()
    
    if existing:
        return {"msg": "Already marked"}
        
    db.add(Attendance(
        student_id=data.student_id,
        date=now.date(),
        timestamp=now,
        session_type=data.session_type,
        status="Manual Override",
        marked_by="Teacher"
    ))
    db.commit()
    return {"msg": "Marked successfully"}

@app.get("/api/export")
def export_excel(db: Session = Depends(get_db)):
    if not HAS_PANDAS:
        return {"msg": "Pandas not installed. Install pandas & openpyxl."}
        
    records = db.query(Attendance, Student).join(Student, Attendance.student_id == Student.id).all()
    
    data = []
    for att, stu in records:
        data.append({
            "Date": att.date.strftime("%Y-%m-%d"),
            "Roll Number": stu.roll_number,
            "Student Name": stu.name,
            "Class": stu.class_section,
            "Session": att.session_type,
            "Status": att.status,
            "Time": att.timestamp.strftime("%I:%M %p"),
            "Marked By": att.marked_by
        })
        
    df = pd.DataFrame(data)
    file_path = "attendance_report.xlsx"
    df.to_excel(file_path, index=False)
    
    return FileResponse(file_path, filename="EduvisionAI_Report.xlsx")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
