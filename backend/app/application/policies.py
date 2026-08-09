from app.application.ports import Authorizer
from app.domain.models import ActorContext, Intent, PolicyChunk


class PolicyAuthorizer(Authorizer):
    async def can_access_intent(self, actor: ActorContext, intent: Intent) -> bool:
        if "employee" not in actor.roles and "hr_admin" not in actor.roles:
            return False
        if intent == Intent.LEAVE_APPLICATION:
            return "employee" in actor.roles
        if intent == Intent.JIRA_TIME_LOG_LOOKUP:
            return "employee" in actor.roles or "manager" in actor.roles
        if intent == Intent.POLICY_QA:
            return True
        return True

    async def can_access_policy_chunk(self, actor: ActorContext, chunk: PolicyChunk) -> bool:
        countries = set(chunk.metadata.get("countries", []))
        departments = set(chunk.metadata.get("departments", []))
        country_ok = not countries or actor.country in countries or "global" in countries
        department_ok = not departments or actor.department in departments or "all" in departments
        return country_ok and department_ok

