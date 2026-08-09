from datetime import datetime, timedelta, timezone
from functools import wraps
import secrets
from uuid import uuid4

from opentelemetry import trace
from pydantic import ValidationError

from app.agent.state import AgentState
from app.application.extraction import extract_leave_fields, extract_time_log_fields, missing_fields
from app.application.guardrails import (
    FaithfulnessPolicy,
    assess_grounding,
    blocked_answer,
    build_citations,
    detect_prompt_injection,
    filter_indirect_prompt_injection,
)
from app.adapters.observability.metrics import (
    AGENT_NODE_SECONDS,
    SECURITY_BLOCKS,
    increment,
    observe_histogram,
)
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
    GroundingReport,
    Intent,
    JiraTimeLogRequest,
    LeaveApplicationRequest,
    PendingAction,
)


class AuthorizationError(RuntimeError):
    pass


class ConfirmationRequired(RuntimeError):
    pass


tracer = trace.get_tracer(__name__)


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
        faithfulness_policy: FaithfulnessPolicy | None = None,
        confirmation_ttl_seconds: int = 900,
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
        self.faithfulness_policy = faithfulness_policy or FaithfulnessPolicy()
        self.confirmation_ttl_seconds = confirmation_ttl_seconds
        self._nodes = (
            self.validate_request,
            self.authenticate,
            self.create_actor_context,
            self.load_thread_checkpoint,
            self.classify_intent,
            self.authorize,
            self.prompt_injection_guardrail,
            self.detect_missing_fields,
            self.retrieve,
            self.rerank,
            self.generate,
            self.faithfulness_guardrail,
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
                state = await self._run_node(node, state)
        response = state["response"]
        if request.idempotency_key:
            await self.idempotency.put(request.idempotency_key, response, ttl_seconds=86400)
            if state.get("idempotency_reserved"):
                await self.idempotency.release(request.idempotency_key)
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
            graph.add_node(node.__name__, self._wrap_node(node))
            if previous is None:
                graph.set_entry_point(node.__name__)
            else:
                graph.add_edge(previous.__name__, node.__name__)
            previous = node
        graph.add_edge(previous.__name__, END)
        return graph.compile()

    def _wrap_node(self, node):
        @wraps(node)
        async def wrapped(state: AgentState) -> AgentState:
            return await self._run_node(node, state)

        return wrapped

    async def _run_node(self, node, state: AgentState) -> AgentState:
        with tracer.start_as_current_span(f"agent.{node.__name__}") as span:
            request = state.get("request")
            actor = state.get("actor")
            intent = state.get("intent")
            span.set_attribute("agent.node", node.__name__)
            span.set_attribute("thread.id", state.get("thread_id", ""))
            if request is not None:
                span.set_attribute("request.has_thread_id", bool(request.thread_id))
            if actor is not None:
                span.set_attribute("actor.roles", ",".join(sorted(actor.roles)))
            if intent is not None:
                span.set_attribute("agent.intent", intent.value)
            import time

            start = time.perf_counter()
            result = await node(state)
            observe_histogram(
                AGENT_NODE_SECONDS,
                {"node": node.__name__, "intent": (result.get("intent") or intent or Intent.UNKNOWN).value},
                time.perf_counter() - start,
            )
            grounding = result.get("answer_grounding")
            if grounding is not None:
                span.set_attribute("grounding.grounded", grounding.grounded)
                span.set_attribute("grounding.faithfulness_score", grounding.faithfulness_score)
                span.set_attribute("grounding.guardrail_action", grounding.guardrail_action)
            if "retrieved_chunks" in result:
                span.set_attribute("rag.retrieved_chunks", len(result.get("retrieved_chunks", [])))
            if "reranked_chunks" in result:
                span.set_attribute("rag.reranked_chunks", len(result.get("reranked_chunks", [])))
            return result

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

    async def load_thread_checkpoint(self, state: AgentState) -> AgentState:
        checkpoint = await self.conversations.load_checkpoint(state["thread_id"])
        if checkpoint is None:
            return state
        actor_subject = checkpoint.get("actor_subject")
        if actor_subject and actor_subject != state["actor"].subject:
            raise AuthorizationError("actor is not authorized for this thread")
        state["previous_checkpoint"] = checkpoint
        return state

    async def classify_intent(self, state: AgentState) -> AgentState:
        state["intent"] = await self.llm.classify_intent(state["request"].message)
        return state

    async def authorize(self, state: AgentState) -> AgentState:
        if not await self.authorizer.can_access_intent(state["actor"], state["intent"]):
            raise AuthorizationError("actor is not authorized for this action")
        return state


    async def prompt_injection_guardrail(self, state: AgentState) -> AgentState:
        reason = detect_prompt_injection(state["request"].message)
        if reason:
            increment(SECURITY_BLOCKS, {"reason": "prompt_injection"})
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state.get("intent", Intent.UNKNOWN),
                answer="I cannot follow instructions that try to override system, security, or tool rules.",
            )
        return state

    async def detect_missing_fields(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        request = state["request"]
        actor = state["actor"]
        if request.confirm:
            await self._resolve_pending_confirmation(state)
            if "response" in state:
                return state
            if state.get("tool_payload"):
                state["missing_fields"] = []
                return state
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

    async def _resolve_pending_confirmation(self, state: AgentState) -> None:
        token = state["request"].confirmation_token
        if not token:
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="I could not confirm that action because the confirmation token is missing.",
            )
            return
        pending = await self.conversations.load_pending_action(token)
        if pending is None:
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="That confirmation has expired or is not valid. Please start the request again.",
            )
            return
        if (
            pending.actor_subject != state["actor"].subject
            or pending.thread_id != state["thread_id"]
            or pending.intent != Intent.LEAVE_APPLICATION
        ):
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="That confirmation does not match this user, thread, or action.",
            )
            return
        if not await self.authorizer.can_access_intent(state["actor"], pending.intent):
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=pending.intent,
                answer="You are not authorized to confirm that action.",
            )
            return
        state["intent"] = pending.intent
        state["tool_payload"] = pending.tool_payload
        state["pending_action_token"] = token

    async def retrieve(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] != Intent.POLICY_QA:
            return state
        chunks = await self.retriever.retrieve(state["request"].message, state["actor"], self.top_k)
        allowed = [
            chunk for chunk in chunks if await self.authorizer.can_access_policy_chunk(state["actor"], chunk)
        ]
        safe_chunks, blocked_chunks = filter_indirect_prompt_injection(allowed)
        if blocked_chunks:
            increment(SECURITY_BLOCKS, {"reason": "indirect_prompt_injection"})
        state["retrieved_chunks"] = safe_chunks
        return state

    async def rerank(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] == Intent.POLICY_QA:
            state["reranked_chunks"] = await self.reranker.rerank(
                state["request"].message, state.get("retrieved_chunks", [])
            )
        return state

    async def generate(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] == Intent.POLICY_QA:
            state["answer"] = await self.llm.answer_policy(
                state["request"].message, state.get("reranked_chunks", [])
            )
        return state

    async def faithfulness_guardrail(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
        if state["intent"] != Intent.POLICY_QA:
            state["answer_grounding"] = GroundingReport(guardrail_action="not_applicable")
            return state
        chunks = state.get("reranked_chunks", [])[: self.top_k]
        answer = state.get("answer", "")
        grounding = assess_grounding(answer, chunks, self.faithfulness_policy)
        state["answer_grounding"] = grounding
        if not grounding.grounded:
            state["answer"] = blocked_answer()
            state["citations"] = []
            return state
        state["citations"] = build_citations(chunks, answer)
        return state

    async def validate_tool(self, state: AgentState) -> AgentState:
        if "response" in state:
            return state
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
                state["tool_payload"] = JiraTimeLogRequest(**state["tool_payload"]).model_dump(mode="json")
            elif state["intent"] == Intent.LEAVE_APPLICATION:
                state["tool_payload"] = LeaveApplicationRequest(**state["tool_payload"]).model_dump(mode="json")
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
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.confirmation_ttl_seconds)
            await self.conversations.save_pending_action(
                PendingAction(
                    token=token,
                    thread_id=state["thread_id"],
                    actor_subject=state["actor"].subject,
                    intent=Intent.LEAVE_APPLICATION,
                    tool_payload=state["tool_payload"],
                    message=state["request"].message,
                    expires_at=expires_at,
                )
            )
            state["response"] = ChatResponse(
                thread_id=state["thread_id"],
                intent=state["intent"],
                answer="Please confirm you want me to submit this leave application.",
                requires_confirmation=True,
                confirmation_token=token,
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
            if state["intent"] in {Intent.JIRA_TIME_LOG_LOOKUP, Intent.LEAVE_APPLICATION}:
                reserved = await self.idempotency.reserve(key, ttl_seconds=120)
                if not reserved:
                    state["response"] = ChatResponse(
                        thread_id=state["thread_id"],
                        intent=state["intent"],
                        answer="That request is already being processed. Please retry with the same idempotency key shortly.",
                    )
                    return state
                state["idempotency_reserved"] = True
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
            if state.get("pending_action_token"):
                await self.conversations.delete_pending_action(state["pending_action_token"])
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
                citations=state.get("citations", build_citations(chunks, state.get("answer", ""))),
                answer_grounding=state.get(
                    "answer_grounding",
                    GroundingReport(guardrail_action="not_applicable"),
                ),
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
            "actor_subject": state["actor"].subject if state.get("actor") else None,
            "employee_id": state["actor"].employee_id if state.get("actor") else None,
            "intent": state["intent"].value if state.get("intent") else None,
            "missing_fields": state.get("missing_fields", []),
            "last_answer": state.get("answer"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
