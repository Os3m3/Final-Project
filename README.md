# 📸 Face Recognition Attendance Management System

## 🧩 Overview
This project is a **Face Recognition–based Attendance Management System** developed as part of the *Final Project (Data Science)*.  
It automates attendance tracking using facial recognition and provides a real-time analytics dashboard for employee management.

The system combines:
- **AI & Computer Vision** (face embedding and recognition)
- **Database integration** (SQLAlchemy ORM)
- **Data visualization** (Streamlit dashboard)
- **Data analytics** (attendance, lateness, and performance insights)

---

## ⚙️ System Architecture

**Flow Overview:**

1. **Face Enrollment (Streamlit – Manage Users)**
   - Users’ names and face images are uploaded.
   - Deep embeddings are generated using `compute_embedding_bgr()` and stored in the database.

2. **Live Recognition (live_engine.py)**
   - The camera captures faces in real-time.
   - The system compares embeddings with stored users.
   - Depending on the time of day (Gate 1–4), the appropriate check-in or checkout is recorded.

3. **Database Storage**
   - Attendance data (check-ins, checkout, date, quarter ID) is saved in the **Attendance** table.
   - The **Users** table stores user details and embeddings.

4. **Streamlit Dashboard (streamlit_app.py)**
   - Displays summarized statistics, KPIs, and charts.
   - Includes **Batch Overview**, **Individual Detail** and **Manage Users** pages for analytics.
   - Provides full user management (Add / Update / Delete).

---

## ✅ Implemented Features (Project Requirements)

| # | Requirement from PDF | Status | Description |
|---|----------------------|--------|--------------|
| 1 | Face Recognition using AI model | ✅ Done | Uses OpenCV + embedding-based recognition |
| 2 | Database integration (CRUD) | ✅ Done | SQLAlchemy ORM with `User` and `Attendance` tables |
| 3 | Check-in and Checkout logic | ✅ Done | Implemented in `gate_logic.py` with four gates |
| 4 | Real-time camera recognition | ✅ Done | Continuous video feed via `Camera` class |
| 5 | Streamlit dashboard for visualization | ✅ Done | Interactive interface for all analytics |
| 6 | Batch overview statistics | ✅ Done | Attendance, absent, and lateness metrics |
| 7 | Individual user analytics | ✅ Done | Per-user attendance records and summaries |
| 8 | Late detection logic | ✅ Done | Auto-detects lateness per gate using time cutoffs |
| 9 | Manage users (Enroll / Update / Delete) | ✅ Done | Add, update, or remove users directly in dashboard |
| 10 | Historical analysis & charts | ✅ Done | Monthly / weekly charts with averages |
| 11 | Attendance calculation (worked hours) | ✅ Done | Calculates worked minutes and converts to hh:mm |
| 12 | Database consistency (quarter ID) | ✅ Done | Auto-computed in `gate_logic.py` |
| 13 | Error handling (no face, cooldown, etc.) | ✅ Done | Gracefully handles errors and cooldowns |
| 14 | Lateness breakdown per gate | ✅ Done | KPIs show late count for Gate 1–4 separately |

---

## 💡 Optional & Extended Features

| Feature | Description |
|----------|--------------|
| **Live face stability detection** | Ensures a face is stable for several frames before logging. |
| **Automatic time zone handling (Asia/Muscat)** | Localized timestamps across all modules. |
| **Streamlit-based User Management** | Full CRUD operations without direct DB access. |
| **Attendance analytics dashboard** | Responsive charts using Plotly. |
| **Dynamic filters** | Select date range, aggregation (weekly/monthly), or specific users. |
| **Cache optimization** | Streamlit caching for faster data retrieval. |
| **Worked hours computation** | Calculates total time between check-ins and checkout. |


---

## 🗂️ Project Structure

```
project/
│
├── src/
│ ├── attendance/
│ │ └── gate_logic.py               # Attendance logic for check-ins and checkout
│ │
│ ├── db/
│ │ ├── base.py                     # Database connection/session setup
│ │ ├── init_db.py                  # Creates database tables (users, attendance)
│ │ ├── models.py                   # Database models for User & Attendance
│ │ └── utils.py                    # Utility functions (e.g., quarter ID)
│ │
│ ├── recognition/
│ │ ├── camera.py                   # Camera capture class for real-time frames
│ │ ├── embeddings.py               # Face embedding computation logic
│ │ ├── enroll.py                   # Face enrollment and registration logic
│ │ └── match.py                    # Face matching and comparison logic
│ │
│ ├── live_engine.py                # Real-time face recognition and attendance system
│ └── streamlit_app.py              # Streamlit dashboard for analytics and user management
│
├── .env                            # Environment variables (DB credentials, paths, etc.)
├── requirements.txt                # Python dependencies
├── SQLQuery1.sql                   # SQL script for manual database checks
└── README.md                       # Project documentation
```

---

## 🧠 Tools and Technologies

| Category | Tools |
|-----------|-------|
| **Programming Language** | Python 3.10+ |
| **Core Libraries** | OpenCV, NumPy, Pandas, Plotly, Streamlit |
| **Database** | SQLite / SQLAlchemy ORM |
| **AI / ML Library** | DeepFace (for embedding extraction), custom embedding pipeline |
| **Visualization** | Streamlit + Plotly |
| **Environment** | `.venv` (virtual environment), Visual Studio Code |


---

## 🚀 How to Run the Project

### 1. Activate Virtual Environment
```bash
.\.venv\Scripts\activate
```

### 2. Run the Live Face Recognition Engine
```bash
python -m src.recognition.live_engine
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

## 🕒 Gate Timings (Asia/Muscat)

| Gate | Purpose | Time Range |
|------|----------|------------|
| Gate 1 | Morning Check-in | 09:00 – 11:59 |
| Gate 2 | Lunch Out | 12:00 – 12:30 |
| Gate 3 | Lunch In | 12:31 – 13:45 |
| Gate 4 | Evening Check-in / Checkout | 13:46 – 23:00 |

---

## 📊 Dashboard Features Summary

### 1. **Batch Overview**
- Total employees, attendance, absences, and late counts.
- Lateness breakdown per gate.
- Weekly / monthly average attendance charts.
- Responsive pie and line visualizations.

### 2. **Individual Detail**
- Per-employee summary (attendance, absence, lateness).
- Daily records (including absents).
- Worked hours for each day.
- Line and pie charts showing attendance trends.

### 3. **Manage Users**
- Enroll new users via photo upload.
- Update name or face embedding.
- Delete users and their attendance records.
- Real-time database update with confirmation prompts.

---

## 📈 Results
✅ Full attendance automation achieved.  
✅ Accurate lateness detection by gate.  
✅ Clean and responsive analytics dashboard.  
✅ End-to-end workflow from enrollment to report generation.

---

## 🔮 Future Improvements
- Cloud deployment (Streamlit Cloud or Azure).  
- Multi-camera support.  
- Role-based admin panel.  
- Email or WhatsApp notifications for absences.  
---

## 👨‍💻 Contributors
**Osama Said Al Zakwani**  
**Mahmood Salim Al Sawwafi**  
