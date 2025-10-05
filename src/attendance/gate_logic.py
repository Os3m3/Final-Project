from datetime import datetime, time
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from src.db.models import Attendance
from src.db.utils import compute_quarter_id

APP_TZ = ZoneInfo("Asia/Muscat")

# Gate windows:
# 1: Morning, 2: Lunch Out, 3: Lunch In, 4: Evening
WINDOWS = {
    1: (time(9,  0), time(11, 59)),  # Morning work
    2: (time(12, 0), time(12, 30)),  # Lunch Out
    3: (time(12, 31), time(13, 45)), # Lunch In
    4: (time(13, 30), time(23, 59)), # Afternoon + Evening (allow until 23:59)
}

def now_local() -> datetime:
    return datetime.now(tz=APP_TZ)

def active_gate(when: datetime) -> int | None:
    t = when.time()
    for gate, (a, b) in WINDOWS.items():
        if a <= t <= b:
            return gate
    return None

# ---- TZ normalization ----------------------------------------------------------
def _to_naive_local(dt: datetime) -> datetime:
    """Convert any datetime to Asia/Muscat and strip tzinfo (store/compare as naive)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(APP_TZ).replace(tzinfo=None)

# --- internal: map gate -> check-in attribute ----------------------------------
def _checkin_attr_for_gate(gate: int) -> str:
    if gate == 1: return "checkin1_time"
    if gate == 2: return "checkin2_time"
    if gate == 3: return "checkin3_time"
    if gate == 4: return "checkin4_time"
    raise ValueError("Invalid gate")

def _all_checkins_present(row: Attendance) -> bool:
    return (
        row.checkin1_time is not None and
        row.checkin2_time is not None and
        row.checkin3_time is not None and
        row.checkin4_time is not None
    )

def upsert_attendance(session: Session, user_id: int, gate: int, when: datetime) -> str:
    """
    First scan in gate N -> set checkinN_time.
    Single end-of-day checkout_time:
        - Prefer: Gate 4 second scan (after checkin4_time).
        - Fallback: if all 4 check-ins exist and checkout_time is empty,
                    any later scan sets checkout_time.
    Returns: "checkin1" | "checkin2" | "checkin3" | "checkin4" | "checkout" | "noop"
    """
    # Normalize incoming timestamp to local-naive and use *only* 'wn' below
    wn = _to_naive_local(when)
    d = wn.date()
    qid = compute_quarter_id(d)

    row = session.query(Attendance).filter(
        Attendance.user_id == user_id,
        Attendance.date == d
    ).one_or_none()

    if row is None:
        row = Attendance(user_id=user_id, date=d, quarter_id=qid)
        session.add(row)

    # Sanitize any legacy tz-aware values stored previously
    for attr in ("checkin1_time", "checkin2_time", "checkin3_time", "checkin4_time", "checkout_time"):
        val = getattr(row, attr)
        if val is not None and getattr(val, "tzinfo", None) is not None:
            setattr(row, attr, _to_naive_local(val))

    action = "noop"

    # 1) Handle check-in for this gate
    in_attr = _checkin_attr_for_gate(gate)
    current_in = getattr(row, in_attr)
    if current_in is None:
        setattr(row, in_attr, wn)
        action = f"checkin{gate}"
    else:
        # 2) Handle single checkout
        if row.checkout_time is None:
            if gate == 4:
                # Preferred: second Gate-4 scan at/after the first one
                if row.checkin4_time is not None and wn >= row.checkin4_time:
                    row.checkout_time = wn
                    action = "checkout"
            else:
                # Fallback: if all check-ins present, any later scan checks out
                if _all_checkins_present(row):
                    latest_in = max(
                        row.checkin1_time,
                        row.checkin2_time,
                        row.checkin3_time,
                        row.checkin4_time,
                    )
                    if wn >= latest_in:
                        row.checkout_time = wn
                        action = "checkout"

    session.commit()
    return action
