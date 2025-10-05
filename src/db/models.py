from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey, LargeBinary, Index
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False, unique=True)
    face_embedding = Column(LargeBinary, nullable=True)  # FaceNet vector as bytes

    attendance_records = relationship(
        "Attendance",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Attendance(Base):
    __tablename__ = "attendance"

    record_id  = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    date       = Column(Date, nullable=False, index=True)
    quarter_id = Column(String(10), nullable=False, index=True)

    # Four check-ins (one for each gate)
    checkin1_time = Column(DateTime, nullable=True)   # Morning
    checkin2_time = Column(DateTime, nullable=True)   # Lunch Out
    checkin3_time = Column(DateTime, nullable=True)   # Lunch In
    checkin4_time = Column(DateTime, nullable=True)   # Evening

    # ✅ Single end-of-day checkout for the whole day
    checkout_time = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="attendance_records")


# one row per (user, date)
Index("ix_attendance_user_date_unique", Attendance.user_id, Attendance.date, unique=True)