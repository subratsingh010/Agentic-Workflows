from uuid import uuid4

from pydantic import ValidationError

from app.agent.state import AgentState
from app.application.extraction import extract_leave_fields, extract_time_log_fields, missing_fields
from app.application.ports import (
    AuditRepository,
    Authenticator,
    Authorizer,
    ConversationRepository,
    IdempotencyStore,
    JiraClient,
    LLMGenerator,
    LeaveClient,
    PolicyRetriever,
    RateLimiter,
    Reranker,
)
from app.domain.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    Intent,
    JiraTimeLogRequest,
    LeaveApplicationRequest,
)


class AuthorizationError(RuntimeError):
    pass


class ConfirmationRequired(RuntimeError):
    pass


class SingleEmployeeSupportAgent:
    def __init__(
        self,
        authenticator: Authenticator,
        authorizer: Authorizer,
        conversations: ConversationRepository,
        audit: AuditRepository,
        idempotency: IdempotencyStore,
        rate_limiter: RateLimiter,
        retriever: PolicyRetriever,
        reranker: Reranker,
        llm: LLMGenerator,
        jira: JiraClient,
        leave: LeaveClient,
        top_k: int,
    ) -> None:
        self.authenticator = authenticator
        self.authorizer = authorizer
        self.conversations = conversations
        self.audit = audit
        self.idempotency = idempotency
        self.rate_limiter = rate_limiter
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.jira = jira
        self.leave = leave
        self.top_k = top_k
        self._nodes = (
            self.validate_request,
            self.authenticate,
            self.create_actor_context,
            self.classify_intent,
            self.authorize,
            self.detect_missing_fields,
            self.retrieve,
            self.rerank,
            self.generate,
            self.validate_tool,
            self.confirmation_guardrail,
            self.idempotency_check,
            self.execute_tool,
            self.audit_log,
            self.respond,
        )
        self.graph = self._build_graph()

    async def run(self, request: ChatRequest, bearer_token: str) -> ChatResponse:
        state: AgentState = {"request": request, "token": bearer_token}
        if self.graph is not None:
            state = await self.graph.ainvoke(state)
        else:
            for node in self._nodes:
                state = await node(state)
        response = state["response"]
        if request.idempotency_key:
            await self.idempotency.put(request.idempotency_key, response, ttl_seconds=86400)
        await self.conversations.save_checkpoint(state["thread_id"], self._serializable_checkpoint(state))
        return response

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(AgentState)
        previous = None
        for node in self._nodes:
            graph.add_node(node.__name__, node)
            if previous is None:
                graph.set_entry_point(node.__name__)
            else:
                graph.add_edge(previous.__name__, node.__name__)
            previous = node
        graph.add_edge(previous.__name__, END)
        return graph.compile()

    async def validate_request(self, state: AgentState) -> AgentState:
        state["thread_id"] = state["request"].thread_id or str(uuid4())
        await self.rate_limiter.check(state["thread_id"])
        return state

    async def authenticate(self, state: AgentState) -> AgentState:
        state["actor"] = await self.authenticator.authenticate(state["token"])
        return state

    async def create_actor_context(self, state: AgentState) -> AgentState:
        actor = state["actor"]
        if not actor.employee_id:
            raise AuthorizationError("authenticated principal has no employee_id")
        return state

    async def classify_intent(self, state: AgentState) -> AgentState:
        state["intent"] = await self.llm.classify_intent(state["request"].message)
        return state

    async def authorize(self, state: AgentState) -> AgentState:
        if not await self.authorizer.can_access_intent(state["actor"], state["intent"]):
            raise AuthorizationError("actor is not authorized for this action")
        return state

    async def detect_missing_fields(self, state: AgentState) -> AgentState:
        request = state["request"]
        actor = state["actor"]
        if state["intent"] == Intent.JIRA_TIME_LOG_LOOKUP:
            payload = extract_time_log_fields(request.message, actor.employee_id)
            state["tool_payload"] = payload
            state["missing_fields"] = missing_fields(payload, ["start_date", "end_date"])
        elif state["intent"] == Intent.LEAVE_APPLICATION:
            payload = extract_leave_fields(request.message, actor.employee_id)
            state["tool_payload"] = payload
            state["missing_fields"] = missing_fields(payload, ["leave_type", "start_date", "end_date"])
        else:
            state["missing_fields"] = []
        return state

    async def retrieve(self, state: AgentState) -> AgentState:
        if state["intent"] != Intent.POLICY_QA:
            return state
        chunks = await self.retriever.retrieve(state["request"].message, state["actor"], self.top_k)
        allowed = [
            chunk for chunk in chunks if await self.authorizer.can_access_policy_chunk(state["actor"], chunk)
        ]
        state["retrieved_chunks"] = allowed
        return state

    async def rerank(self, state: AgentState) -> AgentState:
        if state["intent"] == Intent.POLICY_QA:
            state["reranked_chunks"] = await self.reranker.rerank(
                state["request"].message, state.get("retrieved_chunks", [])
            )
        return state

    async def generate(self, state: AgentState) -> AgentState:
        if state["intent"] == Intent.POLICY_QA:
            state["answer"] = await self.llm.answer_policy(
                state["request"].message, state.get("reranked_chunks", [])
            )
        return state

    async def validate_tool(self, state: AgentState) -> AgentState:
        if state.get("missing_fields"):
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="I need a few more details before I can continue.",
                missing_fields=state["missing_fields"],
            )
            return state
        try:
            if state["intent"] == Intent.JIRA_TIME_LOG_LOOKUP:
                state["tool_payload"] = JiraTimeLogRequest(**state["tool_payload"]).model_dump()
            elif state["intent"] == Intent.LEAVE_APPLICATION:
                state["tool_payload"] = LeaveApplicationRequest(**state["tool_payload"]).model_dump()
        except ValidationError as exc:
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="Some fields were invalid.",
                missing_fields=[error["loc"][0] for error in exc.errors()],
            )
        return state

    async def confirmation_guardrail(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] == Intent.LEAVE_APPLICATION and not state["request"].confirm:
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="Please confirm you want me to submit this leave application.",
                requires_confirmation=True,
            )
        return state

    async def idempotency_check(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        key = state["request"].idempotency_key
        if key:
            existing = await self.idempotency.get(key)
            if existing:
                state["response"] = existing
        return state

    async def execute_tool(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] == Intent.JIRA_TIME_LOG_LOOKUP:
            result = await self.jira.get_time_logs(JiraTimeLogRequest(**state["tool_payload"]))
            state["tool_result"] = result.model_dump(mode="json")
            state["answer"] = f"I found {result.total_hours} Jira hours for that period."
        elif state["intent"] == Intent.LEAVE_APPLICATION:
            result = await self.leave.apply_leave(LeaveApplicationRequest(**state["tool_payload"]))
            state["tool_result"] = result.model_dump(mode="json")
            state["answer"] = f"Your leave request {result.request_id} was submitted."
        elif state["intent"] == Intent.SMALL_TALK:
            state["answer"] = "I’m good. I can help with policy questions, Jira time-log lookups, or leave applications."
        elif state["intent"] == Intent.UNSUPPORTED_TOOL:
            state["answer"] = "I can look up Jira time logs, but I cannot create Jira tickets yet."
        return state

    async def audit_log(self, state: AgentState) -> AgentState:
        await self.audit.record(
            state.get("actor"),
            action=f"agent.{state.get('intent', Intent.UNKNOWN)}",
            outcome="response",
            metadata={"thread_id": state.get("thread_id")},
        )
        return state

    async def respond(self, state: AgentState) -> AgentState:
        if "response" in state:
            response = state["response"]
        else:
            chunks = state.get("reranked_chunks", [])
            response = ChatResponse(
                thread_id=state["thread_id"],
                intent=state.get("intent", Intent.UNKNOWN),
                answer=state.get("answer", "I can help with policy questions, Jira time-log lookups, or leave applications."),
                citations=[
                    Citation(
                        document_id=chunk.document_id,
                        title=chunk.title,
                        chunk_id=chunk.chunk_id,
                        score=chunk.score,
                        excerpt=chunk.text[:240],
                    )
                    for chunk in chunks
                ],
                tool_result=state.get("tool_result"),
            )
        await self.conversations.append_message(
            response.thread_id,
            "user",
            state["request"].message,
            {"intent": response.intent.value},
        )
        await self.conversations.append_message(
            response.thread_id,
            "assistant",
            response.answer,
            {"message_id": str(response.message_id)},
        )
        state["response"] = response
        return state

    def _serializable_checkpoint(self, state: AgentState) -> dict:
        return {
            "thread_id": state.get("thread_id"),
            "intent": state["intent"].value if state.get("intent") else None,
            "missing_fields": state.get("missing_fields", []),
            "last_answer": state.get("answer"),
        }
