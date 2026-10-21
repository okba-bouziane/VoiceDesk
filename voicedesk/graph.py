"""LangGraph conversation logic: intent routing, booking slot filling, appointment management,
RAG answers and human handoff. Every spoken reply is emitted through the LangGraph custom
stream so the LiveKit agent can pipe it to TTS token by token."""

from __future__ import annotations

import datetime as dt
import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .booking_client import AppointmentInfo, BookingTools, SlotInfo, ToolError
from .config import prompt
from .llm import LLMClient, StructuredOutputError, structured_call
from .rag import Hit, is_relevant
from .schemas import AgentAction, BookingSlots, Intent, IntentResult, SlotUpdate

log = logging.getLogger("voicedesk.graph")

CLINIC_TZ = ZoneInfo("Africa/Algiers")
Search = Callable[[str], Awaitable[list[Hit]]]
SPECIALTIES = ["general medicine", "dentistry", "pediatrics", "dermatology"]


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    flow: str  # "none" | "booking" | "cancel" | "list"
    awaiting: str  # "none" | "booking_confirm" | "cancel_confirm"
    slots: dict[str, Any]
    phone: str | None
    candidates: list[dict[str, Any]]
    target: dict[str, Any] | None
    intent: str
    action: dict[str, Any]


@dataclass
class Deps:
    llm: LLMClient
    tools: BookingTools
    search: Search
    today: Callable[[], dt.date] = lambda: dt.datetime.now(CLINIC_TZ).date()
    min_similarity: float = 0.5
    history: int = 8


# ---- speech helpers ---------------------------------------------------------------------------
def calendar_text(today: dt.date) -> str:
    days = (today + dt.timedelta(days=i) for i in range(14))
    return ", ".join(f"{d.strftime('%A')} {d.isoformat()}" for d in days)


def say_date(d: dt.date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def say_time(t: dt.time) -> str:
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def _iso(d: str, t: str) -> tuple[dt.date, dt.time]:
    return dt.date.fromisoformat(d), dt.time.fromisoformat(t)


def say_slot(s: SlotInfo) -> str:
    d, t = _iso(s.date, s.start_time)
    return f"{say_date(d)} at {say_time(t)} with {s.provider_name}"


def say_appt(a: AppointmentInfo) -> str:
    d, t = _iso(a.date, a.start_time)
    return f"{a.specialty} on {say_date(d)} at {say_time(t)} with {a.provider_name}"


def say_digits(phone: str) -> str:
    return " ".join(phone)


def _friendly(err: ToolError) -> str:
    """Tool errors are phrased for logs; rewrite the ones the caller will hear."""
    if err.code == "ALREADY_EXISTS":
        return "that time is no longer available"
    return err.message


def _emit(text: str) -> None:
    get_stream_writer()(text)


def _act(kind: str, speech: str, **kw: Any) -> dict[str, Any]:
    """Validate the final action of the turn and speak it."""
    action = AgentAction(kind=kind, speech=speech, **kw)  # type: ignore[arg-type]
    _emit(action.speech)
    log.info("action %s", action.model_dump(exclude={"speech"}))
    return action.model_dump(mode="json")


# ---- routing ---------------------------------------------------------------------------------
def last_user_text(state: State) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def transcript(state: State, n: int) -> str:
    lines = []
    for m in state.get("messages", [])[-n:]:
        who = "Caller" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def decide(state: State, intent: Intent) -> str:
    """Pure routing decision from (state, intent) to the next node name."""
    flow = state.get("flow", "none")
    awaiting = state.get("awaiting", "none")
    if intent is Intent.HANDOFF:
        return "handoff"
    if awaiting != "none" and intent in (Intent.CONFIRM, Intent.DECLINE):
        if intent is Intent.DECLINE:
            return "decline_flow"
        return "book" if awaiting == "booking_confirm" else "do_cancel"
    if intent is Intent.CANCEL:
        return "manage"
    if intent is Intent.LIST:
        return "manage"
    if intent in (Intent.BOOK, Intent.CHECK):
        return "booking"
    if intent is Intent.INFO:
        return "faq"
    # confirm/decline/other/smalltalk: keep going if a flow is waiting for details
    if flow == "booking" and intent is not Intent.SMALLTALK:
        return "booking"
    if flow in ("cancel", "list") and intent is not Intent.SMALLTALK:
        return "manage"
    return "smalltalk"


def build_graph(deps: Deps):  # type: ignore[no-untyped-def]
    llm, tools = deps.llm, deps.tools

    async def route(state: State) -> dict[str, Any]:
        text = last_user_text(state)
        if not text.strip():
            return {"intent": Intent.OTHER.value, "action": {}}
        system = prompt(
            "router",
            flow=state.get("flow", "none"),
            awaiting=state.get("awaiting", "none") != "none",
        )
        try:
            res = await structured_call(
                llm,
                "route",
                IntentResult,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript(state, 4)},
                ],
            )
            intent = res.intent
        except StructuredOutputError:
            intent = Intent.OTHER
        return {"intent": intent.value, "action": {}}

    def after_route(state: State) -> str:
        if not last_user_text(state).strip() or state.get("action", {}).get("kind") == "error":
            return END
        return decide(state, Intent(state["intent"]))

    async def extract(state: State, known: dict[str, Any]) -> SlotUpdate:
        today = deps.today()
        system = prompt(
            "extract_booking",
            today=today.isoformat(),
            weekday=today.strftime("%A"),
            calendar=calendar_text(today),
            specialties=", ".join(SPECIALTIES),
            known=known or "nothing yet",
        )
        try:
            return await structured_call(
                llm,
                "extract",
                SlotUpdate,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript(state, 4)},
                ],
            )
        except StructuredOutputError:
            return SlotUpdate()

    # ---- booking flow
    async def booking(state: State) -> dict[str, Any]:
        slots = BookingSlots(**(state.get("slots") or {}))
        known = {k: str(v) for k, v in slots.model_dump().items() if v is not None}
        update = await extract(state, known)
        note = ""
        if update.specialty:
            norm = update.specialty.lower()
            match = next((s for s in SPECIALTIES if s in norm or norm in s), None)
            if match is None:
                update.specialty = None
                note = f"We offer {', '.join(SPECIALTIES[:-1])} and {SPECIALTIES[-1]}. "
            else:
                update.specialty = match
        today = deps.today()
        if update.date and update.date < today:
            update.date = None
            note = "That date has already passed. "
        slots = slots.merge(update)
        patch: dict[str, Any] = {"flow": "booking", "awaiting": "none"}

        if slots.specialty and slots.date and slots.time is None:
            try:
                free = await tools.check_availability(slots.specialty, slots.date.isoformat(), 3)
            except ToolError as err:
                slots = slots.model_copy(update={"date": None})
                patch["slots"] = slots.model_dump(mode="json")
                patch["action"] = _act(
                    "ask_slot", f"{err.message}. Which other day would suit you?"
                )
                return patch
            if not free:
                slots = slots.model_copy(update={"date": None})
                patch["slots"] = slots.model_dump(mode="json")
                patch["action"] = _act(
                    "ask_slot",
                    f"{note}There is nothing free that day. Which other day would suit you?",
                )
                return patch
            assert slots.date
            patch["slots"] = slots.model_dump(mode="json")
            times = ", ".join(say_time(_iso(s.date, s.start_time)[1]) for s in free)
            patch["action"] = _act(
                "offer_slots",
                f"{note}On {say_date(slots.date)} I have {times}. Which time do you prefer?",
            )
            return patch

        patch["slots"] = slots.model_dump(mode="json")
        missing = slots.missing()
        if missing:
            questions = {
                "specialty": "Which kind of doctor would you like to see: general medicine, "
                "dentistry, pediatrics or dermatology?",
                "date": "Which day would you like to come?",
                "time": "What time would suit you?",
                "patient_name": "May I have the patient's full name?",
                "patient_phone": "And a phone number to reach you?",
            }
            patch["action"] = _act("ask_slot", note + questions[missing[0]])
            return patch

        assert slots.date and slots.time and slots.specialty
        patch["awaiting"] = "booking_confirm"
        patch["action"] = _act(
            "confirm_booking",
            f"To confirm: {slots.specialty} on {say_date(slots.date)} at {say_time(slots.time)} "
            f"for {slots.patient_name}, phone {say_digits(slots.patient_phone or '')}. "
            "Shall I book it?",
        )
        return patch

    async def book(state: State) -> dict[str, Any]:
        s = BookingSlots(**state["slots"])
        assert s.date and s.time and s.specialty and s.patient_name and s.patient_phone
        try:
            appt = await tools.book(
                specialty=s.specialty,
                date=s.date.isoformat(),
                start_time=s.time.strftime("%H:%M"),
                patient_name=s.patient_name,
                patient_phone=s.patient_phone,
                reason=s.reason or "",
            )
        except ToolError as err:
            if err.code in ("ALREADY_EXISTS", "INVALID_ARGUMENT"):
                kept = s.model_copy(update={"time": None})
                if err.code == "INVALID_ARGUMENT":
                    kept = kept.model_copy(update={"date": None})
                return {
                    "awaiting": "none",
                    "slots": kept.model_dump(mode="json"),
                    "action": _act(
                        "ask_slot",
                        f"Sorry, {_friendly(err)}. Please choose another "
                        f"{'day' if err.code == 'INVALID_ARGUMENT' else 'time'}.",
                    ),
                }
            return {
                "awaiting": "none",
                "flow": "none",
                "slots": {},
                "action": _act(
                    "error",
                    "I could not complete the booking right now. Would you like me to connect "
                    "you with a staff member?",
                ),
            }
        return {
            "flow": "none",
            "awaiting": "none",
            "slots": {},
            "action": _act(
                "booked",
                f"Done. Your {say_appt(appt)} is booked. Is there anything else I can help with?",
                appointment_id=appt.appointment_id,
            ),
        }

    # ---- cancel / list flow
    async def manage(state: State) -> dict[str, Any]:
        intent = Intent(state["intent"])
        flow = (
            "cancel"
            if intent is Intent.CANCEL
            else "list"
            if intent is Intent.LIST
            else state.get("flow", "cancel")
        )
        update = await extract(state, {"phone": state.get("phone")} if state.get("phone") else {})
        phone = update.patient_phone or state.get("phone")
        patch: dict[str, Any] = {"flow": flow, "awaiting": "none", "phone": phone}
        if not phone:
            verb = "cancel" if flow == "cancel" else "look up"
            patch["action"] = _act(
                "ask_slot",
                f"Sure. What phone number is the appointment booked under, so I can {verb} it?",
            )
            return patch
        try:
            appts = await tools.list_appointments(phone)
        except ToolError as err:
            patch["action"] = _act("error", f"{err.message}. Could you repeat the phone number?")
            patch["phone"] = None
            return patch
        if not appts:
            return {
                "flow": "none",
                "phone": None,
                "action": _act(
                    "appointments", "I don't see any upcoming appointments under that number."
                ),
            }
        if flow == "list":
            listing = "; ".join(say_appt(a) for a in appts)
            return {
                "flow": "none",
                "phone": None,
                "action": _act(
                    "appointments",
                    f"You have {len(appts)} appointment{'s' if len(appts) > 1 else ''}: {listing}.",
                ),
            }
        chosen = appts
        if update.appointment_date:
            chosen = [a for a in appts if a.date == update.appointment_date.isoformat()] or appts
        if len(chosen) > 1:
            listing = " or ".join(say_appt(a) for a in chosen)
            patch["action"] = _act(
                "ask_slot", f"You have several: {listing}. Which one should I cancel?"
            )
            return patch
        a = chosen[0]
        patch["awaiting"] = "cancel_confirm"
        patch["target"] = {"id": a.appointment_id, "phone": phone}
        patch["action"] = _act(
            "cancel_confirm",
            f"I found your {say_appt(a)}. Do you want me to cancel it?",
            appointment_id=a.appointment_id,
        )
        return patch

    async def do_cancel(state: State) -> dict[str, Any]:
        t = state["target"] or {}
        reset = {"flow": "none", "awaiting": "none", "phone": None, "target": None}
        try:
            a = await tools.cancel(t["id"], t["phone"])
        except ToolError as err:
            return {**reset, "action": _act("error", f"I could not cancel it: {err.message}.")}
        return {
            **reset,
            "action": _act(
                "cancelled",
                f"Your {say_appt(a)} is cancelled. Anything else I can do for you?",
                appointment_id=a.appointment_id,
            ),
        }

    async def decline_flow(state: State) -> dict[str, Any]:
        return {
            "flow": "none",
            "awaiting": "none",
            "slots": {},
            "phone": None,
            "target": None,
            "action": _act(
                "decline_flow", "No problem, I have dropped that. Anything else I can help with?"
            ),
        }

    # ---- FAQ with RAG
    async def stream_llm(task: str, system: str, state: State) -> str:
        msgs = [{"role": "system", "content": system}]
        for m in state.get("messages", [])[-deps.history :]:
            msgs.append(
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": str(m.content),
                }
            )
        parts: list[str] = []
        async for token in llm.stream(task, msgs):
            parts.append(token)
            _emit(token)
        return "".join(parts)

    async def faq(state: State) -> dict[str, Any]:
        question = last_user_text(state)
        hits = await deps.search(question)
        if not is_relevant(hits, deps.min_similarity):
            return {
                "action": _act(
                    "answer",
                    "I'm not sure about that one. I can connect you with a staff member if you like.",
                )
            }
        context = "\n\n".join(f"[{h.source}] {h.heading}: {h.content}" for h in hits)
        text = await stream_llm("answer", prompt("faq_answer", context=context), state)
        action = AgentAction(
            kind="answer",
            speech=text or "Sorry, I have no answer.",
            sources=sorted({h.source for h in hits}),
            intent=Intent.INFO,
        )
        return {"action": action.model_dump(mode="json")}

    async def smalltalk(state: State) -> dict[str, Any]:
        text = await stream_llm("smalltalk", prompt("smalltalk"), state)
        action = AgentAction(kind="greeting", speech=text or "Hello, how can I help?")
        return {"action": action.model_dump(mode="json")}

    async def handoff(state: State) -> dict[str, Any]:
        return {
            "flow": "none",
            "awaiting": "none",
            "action": _act(
                "handoff",
                "Of course. I am passing you to a member of our staff now. "
                "If this is a medical emergency, please call the emergency number straight away.",
                handoff=True,
            ),
        }

    def safe(fn):  # type: ignore[no-untyped-def]
        """A provider outage must degrade into an apology, never kill the voice session."""

        @functools.wraps(fn)
        async def wrapper(state: State) -> dict[str, Any]:
            try:
                return await fn(state)  # type: ignore[no-any-return]
            except Exception:
                log.exception("node %s failed", fn.__name__)
                return {
                    "intent": Intent.OTHER.value,
                    "action": _act(
                        "error",
                        "Sorry, I am having a technical problem. Please try again in a moment, "
                        "or say staff and I will connect you with a person.",
                    ),
                }

        return wrapper

    g = StateGraph(State)
    for name, fn in [
        ("route", route),
        ("booking", booking),
        ("book", book),
        ("manage", manage),
        ("do_cancel", do_cancel),
        ("decline_flow", decline_flow),
        ("faq", faq),
        ("smalltalk", smalltalk),
        ("handoff", handoff),
    ]:
        g.add_node(name, safe(fn))
    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        after_route,
        [
            "booking",
            "book",
            "manage",
            "do_cancel",
            "decline_flow",
            "faq",
            "smalltalk",
            "handoff",
            END,
        ],
    )
    for name in (
        "booking",
        "book",
        "manage",
        "do_cancel",
        "decline_flow",
        "faq",
        "smalltalk",
        "handoff",
    ):
        g.add_edge(name, END)
    return g.compile(checkpointer=MemorySaver())
