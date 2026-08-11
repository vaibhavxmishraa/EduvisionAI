# EduvisionAI
👁️ EduvisionAI - Smart CCTV Attendance System

Developed by Tensor Titans

An automated multi-face recognition AI system that streams classroom CCTV feeds, logs twice-a-day attendance (Check-In / Check-Out), triggers real-time SMS notifications to parents via Twilio, and provides a sleek dashboard for Admins, Teachers, and Parents.

✨ Features

🎥 Simultaneous Multi-Face CCTV Detection: Detects multiple students in a single frame using face_recognition and OpenCV.

🛡️ Twice-a-Day Validation Engine: Automatically categorizes logs into Morning Check-In (< 12:00 PM) and Evening Check-Out (>= 12:00 PM), avoiding duplicate logging.

📱 Instant Parent Alerts: Sends SMS alerts to parents when their child arrives at or leaves school via Twilio.

📋 Teacher Roster & Manual Override: Allows teachers to manually mark check-in or check-out if a student is missed by the CCTV.

📊 Automated Excel Exports: Generates downloadable structured .xlsx attendance reports powered by pandas and openpyxl.

⚡ Hardware Fallback / Self-Healing: Includes simulation modes so the frontend and backend can run seamlessly even without a physical camera or heavy AI libraries installed.

📁 Repository Directory Structure

EduvisionAI/
│
├── index.html         # Single-Page Application (Frontend Dashboard)
├── main.py            # FastAPI Backend Server & AI Engine
├── requirements.txt   # Python Dependencies
├── .gitignore         # Git ignore rules for SQLite & Virtual Env
└── README.md          # Project documentation


🚀 Quick Setup & Installation

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/eduvision-ai.git
cd eduvision-ai


2. Set Up Virtual Environment & Install Dependencies

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt


(Note: On Windows/macOS, installing face-recognition requires cmake and a C++ compiler).

3. Run the Backend Server

python main.py


The server will start at http://127.0.0.1:8000 and automatically seed initial demo accounts into eduvision.db.

4. Open the Web Dashboard

Double-click index.html or open it in any modern web browser.

🔑 Default Credentials

Role

Username

Password

Access Level

Admin

admin

password

Live CCTV Feed, Webcam Face Vector Enrollment, Excel Export

Teacher

teacher

password

Classroom Roster, Manual Check-In/Check-Out, Excel Export

Parent

parent

password

Real-time Child Timeline & School Arrival/Departure Logs

📡 API Endpoints Summary

POST /api/auth/login - Authenticates user and returns role profile.

GET /api/camera/stream - Live MJPEG video stream with face recognition bounding boxes.

POST /api/admin/enroll - Captures webcam snapshot and registers 128-d face encoding vectors.

GET /api/attendance/recent - Polls live detection events for toast alerts.

GET /api/parent/{parent_id}/dashboard - Returns timeline notifications and child location status.

GET /api/teacher/students - Returns student roster with check-in/out statuses.

POST /api/teacher/mark - Manual attendance override endpoint.

GET /api/export - Downloads structured .xlsx attendance report file.

🛠️ Built With

Frontend: HTML5, Tailwind CSS, Vanilla JavaScript

Backend: FastAPI, SQLAlchemy, SQLite, Pydantic

AI & Video Processing: OpenCV (cv2), face_recognition, numpy

SMS & Reporting: Twilio API, Pandas, OpenPyXL

👥 Credits & Team

Designed & Built by Tensor Titans.
