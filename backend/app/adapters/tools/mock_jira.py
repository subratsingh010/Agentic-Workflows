from datetime import timedelta

from app.application.ports import JiraClient
from app.domain.models import JiraTimeLogEntry, JiraTimeLogRequest, JiraTimeLogResponse


class MockJiraClient(JiraClient):
    async def get_time_logs(self, request: JiraTimeLogRequest) -> JiraTimeLogResponse:
        entries = [
            JiraTimeLogEntry(
                issue_key="SUP-101",
                summary="Employee portal support",
                hours=3.5,
                work_date=request.start_date,
            ),
            JiraTimeLogEntry(
                issue_key="PLAT-88",
                summary="Platform reliability work",
                hours=4.0,
                work_date=min(request.end_date, request.start_date + timedelta(days=1)),
            ),
        ]
        return JiraTimeLogResponse(
            employee_id=request.employee_id,
            entries=entries,
            total_hours=sum(entry.hours for entry in entries),
        )

