import io
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Your project imports
from src.db.base import get_session
from src.db.models import User, Attendance

# DeepFace (optional tab)
try:
    from deepface import DeepFace
    _HAS_DEEPFACE = True
except Exception:
    _HAS_DEEPFACE = False


# ---------------------------
# Helpers: DB -> DataFrames
# ---------------------------
@st.cache_data(ttl=60)
def load_attendance() -> pd.DataFrame:
    """Load joined attendance records with user names into a DataFrame."""
    with get_session() as s:
        engine = s.get_bind()
        q = """
        SELECT
            a.record_id,
            a.user_id,
            u.name AS user_name,
            CAST(a.date AS DATE) AS date,
            a.quarter_id,
            a.checkin1_time,
            a.checkin2_time,
            a.checkin3_time,
            a.checkin4_time,
            a.checkout_time
        FROM attendance AS a
        INNER JOIN users AS u ON u.user_id = a.user_id
        """
        df = pd.read_sql(q, engine)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["checkin1_time","checkin2_time","checkin3_time","checkin4_time","checkout_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(ttl=60)
def load_users() -> pd.DataFrame:
    with get_session() as s:
        engine = s.get_bind()
        df = pd.read_sql("SELECT user_id, name FROM users ORDER BY name;", engine)
    return df


def compute_compliance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for per-gate present flags and total gates completed."""
    if df.empty:
        return df.assign(
            present_g1=0, present_g2=0, present_g3=0, present_g4=0, gates_completed=0
        )
    out = df.copy()
    out["present_g1"] = out["checkin1_time"].notna().astype(int)
    out["present_g2"] = out["checkin2_time"].notna().astype(int)
    out["present_g3"] = out["checkin3_time"].notna().astype(int)
    out["present_g4"] = out["checkin4_time"].notna().astype(int)
    out["gates_completed"] = out[["present_g1","present_g2","present_g3","present_g4"]].sum(axis=1)
    return out


def worked_minutes(row) -> Optional[int]:
    """
    Compute total worked minutes robustly:
      • Morning:   checkin1_time -> checkin2_time  (if both present)
      • Afternoon: preferred  checkin3_time -> checkout_time
                   fallback   checkin4_time -> checkout_time (if checkin3 missing)
                   else       checkin3_time -> checkin4_time (if no checkout yet)
    """
    try:
        total = 0

        # Morning block
        if pd.notna(row["checkin1_time"]) and pd.notna(row["checkin2_time"]):
            total += int((row["checkin2_time"] - row["checkin1_time"]).total_seconds() // 60)

        # Afternoon / evening block
        if pd.notna(row["checkout_time"]):
            if pd.notna(row["checkin3_time"]):
                total += int((row["checkout_time"] - row["checkin3_time"]).total_seconds() // 60)
            elif pd.notna(row["checkin4_time"]):
                total += int((row["checkout_time"] - row["checkin4_time"]).total_seconds() // 60)
        else:
            # No checkout yet: if both afternoon check-ins exist, count that span
            if pd.notna(row["checkin3_time"]) and pd.notna(row["checkin4_time"]):
                total += int((row["checkin4_time"] - row["checkin3_time"]).total_seconds() // 60)

        return total if total > 0 else None
    except Exception:
        return None


# ---------------------------
# UI: Sidebar / App config
# ---------------------------
st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.sidebar.title("Attendance Dashboard")

# Optional auto-refresh every 30s if available
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30 * 1000, key="data_refresh")
except Exception:
    # If the package isn't installed, we silently skip auto-refresh.
    pass

# Manual one-click cache clear (always available)
if st.sidebar.button("🔄 Refresh data now"):
    try:
        load_attendance.clear()
        load_users.clear()
    except Exception:
        pass
    st.cache_data.clear()
    st.rerun()

view = st.sidebar.radio(
    "Choose a view",
    ["📊 Batch Overview", "👤 Individual Detail", "⚙️ Manage Users"]
)

df_att = load_attendance()
df_users = load_users()

# Derive flags & minutes
df_att = compute_compliance_columns(df_att)
if not df_att.empty:
    df_att["worked_minutes"] = df_att.apply(worked_minutes, axis=1)



# ---------------------------
# 📊 Batch Overview
# ---------------------------
if view == "📊 Batch Overview":
    st.title("📊 Batch Overview")

    # ---- FILTERS
    col1, col2, col3 = st.columns([1,1,1])
    min_date = pd.to_datetime(df_att["date"]).min() if not df_att.empty else pd.Timestamp.today().date()
    max_date = pd.to_datetime(df_att["date"]).max() if not df_att.empty else pd.Timestamp.today().date()
    start = col1.date_input("Start date", value=min_date)
    end = col2.date_input("End date", value=max_date)
    agg_by = col3.selectbox("Aggregate by", ["Week", "Month"], index=1)

    # ---- filter the data
    mask = (pd.to_datetime(df_att["date"]).dt.date >= start) & (pd.to_datetime(df_att["date"]).dt.date <= end)
    dff = df_att.loc[mask].copy()
    if dff.empty:
        st.info("No attendance in the selected date range.")
        st.stop()

    # ---- lateness cutoffs (edit if policy changes)
    CUT_G1 = pd.to_datetime("09:00").time()   # Morning
    CUT_G2 = pd.to_datetime("12:00").time()   # Lunch Out
    CUT_G3 = pd.to_datetime("13:30").time()   # Lunch In
    CUT_G4 = pd.to_datetime("17:00").time()   # Evening

    def _late_gate(ts, cutoff):
        if pd.isna(ts): 
            return 0
        return 1 if ts.time() > cutoff else 0

    # per-gate late flags + any-late
    dff["late_g1"] = dff["checkin1_time"].apply(lambda x: _late_gate(x, CUT_G1))
    dff["late_g2"] = dff["checkin2_time"].apply(lambda x: _late_gate(x, CUT_G2))
    dff["late_g3"] = dff["checkin3_time"].apply(lambda x: _late_gate(x, CUT_G3))
    dff["late_g4"] = dff["checkin4_time"].apply(lambda x: _late_gate(x, CUT_G4))
    dff["late_any"] = dff[["late_g1","late_g2","late_g3","late_g4"]].max(axis=1)

    # ---- daily summary (one row per calendar day)
    dff["_date"] = pd.to_datetime(dff["date"]).dt.date
    all_users_count = int(df_users.shape[0]) if not df_users.empty else dff["user_id"].nunique()

    daily = (
        dff.groupby("_date")
           .agg(
               attendees=("user_id", "nunique"),   # users with at least one check-in that day
               late_users=("late_any", "sum")
           )
           .reset_index()
    )
    daily["absent"] = daily["attendees"].apply(lambda a: max(all_users_count - int(a), 0))

    # ======================
    # KPIs — TOTALS
    # ======================
    total_attend = int(daily["attendees"].sum())
    total_absent = int(daily["absent"].sum())

    # late totals per gate and overall
    total_late_g1 = int(dff["late_g1"].sum())
    total_late_g2 = int(dff["late_g2"].sum())
    total_late_g3 = int(dff["late_g3"].sum())
    total_late_g4 = int(dff["late_g4"].sum())
    total_late_all = total_late_g1 + total_late_g2 + total_late_g3 + total_late_g4

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Employees", f"{all_users_count}")
    k2.metric("Attended (total)", f"{total_attend}")
    k3.metric("Absent (total)", f"{total_absent}")
    k4.metric("Late (total)", f"{total_late_all}")

    # Late per gate (TOTALS + share of all late)
    g1, g2, g3, g4 = st.columns(4)
    for label, val, container in [
        ("Gate 1 late (total)", total_late_g1, g1),
        ("Gate 2 late (total)", total_late_g2, g2),
        ("Gate 3 late (total)", total_late_g3, g3),
        ("Gate 4 late (total)", total_late_g4, g4),
    ]:
        pct = (val / total_late_all * 100.0) if total_late_all > 0 else 0.0
        container.metric(label, f"{val}", delta=f"{pct:.1f}% of late")

    # ======================
    # Charts — AVERAGES (per day)
    # ======================
    day_counts = daily.copy()
    day_counts["_date_ts"] = pd.to_datetime(day_counts["_date"])
    if agg_by == "Week":
        day_counts["_bucket"] = day_counts["_date_ts"] - pd.to_timedelta(day_counts["_date_ts"].dt.dayofweek, unit="d")
        bucket_title = "Weekly"
    else:
        day_counts["_bucket"] = day_counts["_date_ts"].dt.to_period("M").dt.to_timestamp()
        bucket_title = "Monthly"

    per = (day_counts
           .groupby("_bucket")
           .agg(
               avg_attend=("attendees", "mean"),
               avg_absent=("absent", "mean"),
               avg_late=("late_users", "mean"),
           )
           .reset_index())

    avg_attend_overall = float(day_counts["attendees"].mean())
    avg_absent_overall = float(day_counts["absent"].mean())
    avg_late_overall   = float(day_counts["late_users"].mean())

    c1, c2 = st.columns([2,1])

    with c1:
        fig_line = px.line(
            per, x="_bucket", y=["avg_attend","avg_absent"],
            markers=True,
            labels={"_bucket":"Period","value":"People (avg/day)","variable":"Metric"},
            title=f"{bucket_title} Average Attended vs Absent"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        pie_df = pd.DataFrame({
            "label": ["Attended (avg/day)","Absent (avg/day)","Late (avg/day)"],
            "value": [avg_attend_overall, avg_absent_overall, avg_late_overall]
        })
        fig_pie = px.pie(pie_df, names="label", values="value", title="Average Distribution")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)



# ---------------------------
# 👤 Individual Detail (per-user view, with absents in table)
# ---------------------------
elif view == "👤 Individual Detail":
    st.title("👤 Individual Detail")

    if df_users.empty:
        st.warning("No users found.")
        st.stop()

    # ---- Select user
    user_name = st.selectbox("Select user", df_users["name"].tolist())
    user_id = int(df_users.loc[df_users["name"] == user_name, "user_id"].iloc[0])

    # ---- Slice attendance for this user (before date filtering)
    user_all = df_att[df_att["user_id"] == user_id].copy()
    if user_all.empty:
        st.info("No attendance recorded for this user yet.")
        st.stop()

    # ---- Filters (Start / End / Aggregate by)
    c1, c2, c3 = st.columns([1,1,1])
    min_date_u = pd.to_datetime(user_all["date"]).min().date()
    max_date_u = pd.to_datetime(user_all["date"]).max().date()
    start = c1.date_input("Start date", value=min_date_u)
    end   = c2.date_input("End date", value=max_date_u)
    agg_by = c3.selectbox("Aggregate by", ["Week", "Month"], index=1)

    # ---- Filtered frame for this user
    dff = user_all[(pd.to_datetime(user_all["date"]).dt.date >= start) &
                   (pd.to_datetime(user_all["date"]).dt.date <= end)].copy()
    if dff.empty:
        st.info("No records in selected range for this user.")
        st.stop()

    # ---- Lateness cutoffs (same as Batch Overview)
    CUT_G1 = pd.to_datetime("09:00").time()   # Morning
    CUT_G2 = pd.to_datetime("12:00").time()   # Lunch Out
    CUT_G3 = pd.to_datetime("13:30").time()   # Lunch In
    CUT_G4 = pd.to_datetime("17:00").time()   # Evening

    def _late_gate(ts, cutoff):
        if pd.isna(ts):
            return 0
        return 1 if ts.time() > cutoff else 0

    dff["late_g1"] = dff["checkin1_time"].apply(lambda x: _late_gate(x, CUT_G1))
    dff["late_g2"] = dff["checkin2_time"].apply(lambda x: _late_gate(x, CUT_G2))
    dff["late_g3"] = dff["checkin3_time"].apply(lambda x: _late_gate(x, CUT_G3))
    dff["late_g4"] = dff["checkin4_time"].apply(lambda x: _late_gate(x, CUT_G4))
    dff["late_any"] = dff[["late_g1","late_g2","late_g3","late_g4"]].max(axis=1)

    # ---- Build calendar (with weekend toggle)
    include_weekends = st.checkbox("Include weekends", value=True)

    all_days = pd.date_range(start, end, freq="D")
    if include_weekends:
        days = all_days
    else:
        days = all_days[~all_days.weekday.isin([5, 6])]  # Mon–Fri only

    cal = pd.DataFrame({"_date_ts": days})
    cal["_date"] = cal["_date_ts"].dt.date

    # Per-day presence/late flags from actual records
    dff["_date"] = pd.to_datetime(dff["date"]).dt.date
    day_user = (
        dff.groupby("_date")
           .agg(
               present=("record_id", lambda s: 1),     # any record that day => present
               late_users=("late_any", "max")          # any late that day => 1
           )
           .reset_index()
    )

    # Join onto calendar; missing days => absent
    daily = cal.merge(day_user, how="left", on="_date")
    daily["present"] = daily["present"].fillna(0).astype(int)
    daily["late_users"] = daily["late_users"].fillna(0).astype(int)
    daily["absent"] = 1 - daily["present"]

    # ======================
    # KPIs — TOTALS (per user)
    # ======================
    total_attend_days = int(daily["present"].sum())
    total_absent_days = int(daily["absent"].sum())

    total_late_g1 = int(dff["late_g1"].sum())
    total_late_g2 = int(dff["late_g2"].sum())
    total_late_g3 = int(dff["late_g3"].sum())
    total_late_g4 = int(dff["late_g4"].sum())
    total_late_all = total_late_g1 + total_late_g2 + total_late_g3 + total_late_g4

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("User", user_name)
    k2.metric("Attended (total days)", f"{total_attend_days}")
    k3.metric("Absent (total days)", f"{total_absent_days}")
    k4.metric("Late (total events)", f"{total_late_all}")

    g1, g2, g3, g4 = st.columns(4)
    for label, val, container in [
        ("Gate 1 late (total)", total_late_g1, g1),
        ("Gate 2 late (total)", total_late_g2, g2),
        ("Gate 3 late (total)", total_late_g3, g3),
        ("Gate 4 late (total)", total_late_g4, g4),
    ]:
        pct = (val / total_late_all * 100.0) if total_late_all > 0 else 0.0
        container.metric(label, f"{val}", delta=f"{pct:.1f}% of late")

    # ======================
    # Charts — AVERAGES (per day)
    # ======================
    if agg_by == "Week":
        daily["_bucket"] = daily["_date_ts"] - pd.to_timedelta(daily["_date_ts"].dt.dayofweek, unit="d")  # Monday
        bucket_title = "Weekly"
    else:
        daily["_bucket"] = daily["_date_ts"].dt.to_period("M").dt.to_timestamp()
        bucket_title = "Monthly"

    per = (
        daily.groupby("_bucket")
             .agg(
                 avg_attend=("present", "mean"),
                 avg_absent=("absent", "mean"),
                 avg_late=("late_users", "mean"),
             )
             .reset_index()
    )

    avg_attend_overall = float(daily["present"].mean())
    avg_absent_overall = float(daily["absent"].mean())
    avg_late_overall   = float(daily["late_users"].mean())

    c1, c2 = st.columns([2,1])

    with c1:
        fig_line = px.line(
            per, x="_bucket", y=["avg_attend","avg_absent"],
            markers=True,
            labels={"_bucket":"Period","value":"(avg/day)","variable":"Metric"},
            title=f"{bucket_title} Average Attended vs Absent"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        pie_df = pd.DataFrame({
            "label": ["Attended (avg/day)","Absent (avg/day)","Late (avg/day)"],
            "value": [avg_attend_overall, avg_absent_overall, avg_late_overall]
        })
        fig_pie = px.pie(pie_df, names="label", values="value", title="Average Distribution")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ======================
    # Detailed records table (including ABSENT days)
    # ======================
    st.subheader("Detailed Records (including absents)")

    merged = daily.merge(
        dff[[
            "date","checkin1_time","checkin2_time",
            "checkin3_time","checkin4_time","checkout_time","worked_minutes"
        ]],
        how="left",
        left_on="_date",
        right_on="date"
    )

    def _fmt_hhmm(m):
        if pd.isna(m): return "-"
        m = int(m)
        return f"{m//60}h {m%60:02d}m"

    merged["worked_hhmm"] = merged["worked_minutes"].map(_fmt_hhmm)
    merged["Status"] = merged["present"].apply(lambda x: "Present" if x == 1 else "Absent")

    show_cols = [
        "_date","Status","checkin1_time","checkin2_time",
        "checkin3_time","checkin4_time","checkout_time","worked_hhmm"
    ]
    tbl = merged[show_cols].sort_values("_date")
    st.dataframe(tbl, use_container_width=True)


# ---------------------------
#   Manage Users (Enroll / Update / Delete)
# ---------------------------
else:
    st.title("⚙️ Manage Users")

    import pickle, cv2, numpy as np, pandas as pd
    from src.db.base import get_session
    from src.db.models import User, Attendance
    from src.recognition.embeddings import compute_embedding_bgr

    def _bytes_to_bgr(file_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        return img

    @st.cache_data(ttl=30)
    def _load_users_df() -> pd.DataFrame:
        with get_session() as s:
            engine = s.get_bind()
            return pd.read_sql(
                "SELECT user_id, name, CASE WHEN face_embedding IS NULL THEN 0 ELSE 1 END AS has_embedding "
                "FROM users ORDER BY name;", engine
            )

    tab_enroll, tab_update, tab_delete = st.tabs(["➕ Enroll", "✏️ Update", "🗑️ Delete"])

    # ========= ENROLL =========
    with tab_enroll:
        st.subheader("Enroll new user")
        with st.form("enroll_form", clear_on_submit=False):
            name = st.text_input("Full name")
            files = st.file_uploader(
                "Upload one or more face images (JPG/PNG). We’ll average their embeddings.",
                type=["jpg","jpeg","png"], accept_multiple_files=True, key="enroll_imgs"
            )
            submitted = st.form_submit_button("Enroll")

        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
            elif not files:
                st.error("Please upload at least one image.")
            else:
                embs, fail_count = [], 0
                with st.spinner("Processing images and computing embeddings..."):
                    for f in files:
                        try:
                            bgr = _bytes_to_bgr(f.read())
                            if bgr is None:
                                fail_count += 1; continue
                            emb = compute_embedding_bgr(bgr).astype(np.float32)
                            n = np.linalg.norm(emb); emb = emb / n if n > 0 else emb
                            embs.append(emb)
                        except Exception:
                            fail_count += 1
                if not embs:
                    st.error("No valid face detected. Try clearer, single-face photos.")
                else:
                    avg_emb = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
                    emb_bytes = pickle.dumps(avg_emb, protocol=pickle.HIGHEST_PROTOCOL)
                    try:
                        with get_session() as s:
                            u = s.query(User).filter(User.name == name).one_or_none()
                            if u is None:
                                s.add(User(name=name, face_embedding=emb_bytes))
                                s.commit()
                                st.success(f"✅ Enrolled new user: **{name}**  (OK: {len(embs)}, failed: {fail_count})")
                            else:
                                u.face_embedding = emb_bytes
                                s.commit()
                                st.success(f"✅ Updated existing user: **{name}**  (OK: {len(embs)}, failed: {fail_count})")
                        st.dataframe(_load_users_df(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Database error while saving user: {e}")

    # ========= UPDATE =========
    with tab_update:
        st.subheader("Update existing user")
        users_df = _load_users_df()
        if users_df.empty:
            st.info("No users found. Enroll someone first.")
        else:
            pick_name = st.selectbox("Select user", users_df["name"].tolist(), key="upd_pick")
            picked = users_df.loc[users_df["name"] == pick_name].iloc[0]
            user_id = int(picked["user_id"])
            new_name = st.text_input("New name (optional)", value=pick_name, key="upd_newname")

            st.markdown("Upload **zero or more** images:")
            upd_files = st.file_uploader(
                "Upload images (optional, JPG/PNG)", type=["jpg","jpeg","png"],
                accept_multiple_files=True, key="upd_imgs"
            )

            if st.button("Apply Update", type="primary"):
                try:
                    with get_session() as s:
                        u = s.query(User).filter(User.user_id == user_id).one_or_none()
                        if u is None:
                            st.error("User not found."); st.stop()

                        # Rename
                        if new_name.strip() and new_name.strip() != u.name:
                            name_taken = s.query(User).filter(User.name == new_name.strip(), User.user_id != user_id).one_or_none()
                            if name_taken:
                                st.error("Another user already has that name."); st.stop()
                            u.name = new_name.strip()

                        # Recompute embedding if new images provided
                        if upd_files:
                            embs, fail_count = [], 0
                            with st.spinner("Computing new embedding..."):
                                for f in upd_files:
                                    try:
                                        bgr = _bytes_to_bgr(f.read())
                                        if bgr is None:
                                            fail_count += 1; continue
                                        emb = compute_embedding_bgr(bgr).astype(np.float32)
                                        n = np.linalg.norm(emb); emb = emb / n if n > 0 else emb
                                        embs.append(emb)
                                    except Exception:
                                        fail_count += 1
                            if embs:
                                avg_emb = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
                                u.face_embedding = pickle.dumps(avg_emb, protocol=pickle.HIGHEST_PROTOCOL)
                                st.success(f"✅ Updated embedding for **{u.name}** (OK: {len(embs)}, failed: {fail_count})")
                        s.commit()
                    st.success("Update saved.")
                    st.dataframe(_load_users_df(), use_container_width=True)
                except Exception as e:
                    st.error(f"Database error while updating user: {e}")

    # ========= DELETE =========
    with tab_delete:
        st.subheader("Delete user")
        users_df = _load_users_df()
        if users_df.empty:
            st.info("No users to delete.")
        else:
            pick_name_del = st.selectbox("Select user to delete", users_df["name"].tolist(), key="del_pick")
            picked_del = users_df.loc[users_df["name"] == pick_name_del].iloc[0]
            del_user_id = int(picked_del["user_id"])

            also_delete_att = st.checkbox("Also delete this user's attendance records.", value=True)
            confirm_text = st.text_input("Type the user's name to confirm", value="", key="confirm_name")
            delete_btn = st.button("Delete User", type="secondary")

            if delete_btn:
                if confirm_text.strip() != pick_name_del.strip():
                    st.error("Confirmation text does not match the user's name.")
                else:
                    try:
                        with get_session() as s:
                            if also_delete_att:
                                s.query(Attendance).filter(Attendance.user_id == del_user_id).delete(synchronize_session=False)
                            u = s.query(User).filter(User.user_id == del_user_id).one_or_none()
                            if u: s.delete(u)
                            s.commit()
                        st.success(f"🗑️ Deleted user **{pick_name_del}**")
                        st.dataframe(_load_users_df(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Database error while deleting user: {e}")

    st.caption("💡 Manage enrollments: Add new users, update names or embeddings, or delete users safely (with attendance if needed).")

# Build tag (for sanity check you’re on the latest file)
st.caption("build: streamlit_dashboard v3 • weekend toggle + auto/quick refresh")
