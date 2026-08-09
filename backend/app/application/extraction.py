from datetime import date
import re
from typing import Any


DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def extract_dates(message: str) -> list[date]:
    dates: list[date] = []
    for raw in DATE_RE.findall(message):
        try:
            dates.append(date.fromisoformat(raw))
        except ValueError:
            continue
    return dates


def extract_leave_fields(message: str, employee_id: str) -> dict[str, Any]:
    lowered = message.lower()
    leave_type = None
    for candidate in ("vacation", "sick", "personal"):
        if candidate in lowered:
            leave_type = candidate
            break
    dates = extract_dates(message)
    return {
        "employee_id": employee_id,
        "leave_type": leave_type,
        "start_date": dates[0] if len(dates) >= 1 else None,
        "end_date": dates[1] if len(dates) >= 2 else None,
        "reason": message[:500],
    }


def extract_time_log_fields(message: str, employee_id: str) -> dict[str, Any]:
    dates = extract_dates(message)
    return {
        "employee_id": employee_id,
        "start_date": dates[0] if len(dates) >= 1 else None,
        "end_date": dates[1] if len(dates) >= 2 else None,
    }


def missing_fields(payload: dict[str, Any], required: list[str]) -> list[str]:
    return [field for field in required if payload.get(field) in (None, "")]

