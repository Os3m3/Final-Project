from datetime import date

def compute_quarter_id(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"