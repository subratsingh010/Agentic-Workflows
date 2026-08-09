from datetime import datetime, timezone
from uuid import uuid4

from app.application.ports import LeaveClient
from app.domain.models import LeaveApplicationRequest, LeaveApplicationResponse


class MockLeaveClient(LeaveClient):
    async def apply_leave(self, request: LeaveApplicationRequest) -> LeaveApplicationResponse:
        return LeaveApplicationResponse(
            request_id=f"LV-{uuid4().hex[:10].upper()}",
            status="submitted",
            submitted_at=datetime.now(timezone.utc),
        )

