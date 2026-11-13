"""LangGraph conversation logic: intent routing, booking slot filling, appointment management,
RAG answers and human handoff. Every spoken reply is emitted through the LangGraph custom
stream so the LiveKit agent can pipe it to TTS token by token."""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, TypeVar, cast

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from .booking_client import AppointmentInfo, BookingTools, SlotInfo, ToolError
from .config import CLINIC_TZ, prompt
from .llm import LLMClient, StructuredOutputError, structured_call
from .rag import Hit, is_relevant
from .schemas import ActionKind, AgentAction, BookingSlots, Intent, IntentResult, SlotUpdate

log = logging.getLogger("voicedesk.graph")

Search = Callable[[str], Awaitable[list[Hit]]]
NodeFn = TypeVar("NodeFn", bound=Callable[..., Awaitable[dict[str, Any]]])
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


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_EXPLICIT_DATE = re.compile(
    r"\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b"
    r"|\d{1,2}[/.-]\d{1,2}|\d{4}-\d{2}-\d{2}|after next|following|in two weeks|the week after",
    re.I,
)


def resolve_weekday(text: str, today: dt.date, model_date: dt.date | None) -> dt.date | None:
    """Make "Sunday" / "this Sunday" / "next Sunday" mean the first Sunday after today.

    Models disagree on "next Sunday" (the coming one, or the one after). When the caller's
    message names exactly one weekday and no explicit date, and the model picked that weekday,
    the date is set deterministically. Anything else keeps the model's date; the read-back
    before booking remains the final check.
    """
    if model_date is None:
        return None
    words = re.findall(r"[a-z]+", text.lower())
    named = {w for w in words if w in WEEKDAYS}
    if len(named) != 1 or _EXPLICIT_DATE.search(text):
        return model_date
    target = WEEKDAYS.index(named.pop())
    if model_date.weekday() != target:
        return model_date
    ahead = (target - today.weekday()) % 7 or 7
    return today + dt.timedelta(days=ahead)


def say_date(d: dt.date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def say_time(t: dt.time) -> str:
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def _parse(date: str, start_time: str) -> tuple[dt.date, dt.time]:
    return dt.date.fromisoformat(date), dt.time.fromisoformat(start_time)


def say_slot_time(s: SlotInfo) -> str:
    return say_time(_parse(s.date, s.start_time)[1])


OFFERED_TIMES = 3


def distinct_times(free: list[SlotInfo], n: int = OFFERED_TIMES) -> list[str]:
    """The first n start times, each said once even when several doctors are free then."""
    times: list[str] = []
    for slot in free:
        spoken = say_slot_time(slot)
        if spoken not in times:
            times.append(spoken)
    return times[:n]


def say_appt(a: AppointmentInfo) -> str:
    d, t = _parse(a.date, a.start_time)
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


def _act(kind: ActionKind, speech: str, **kw: Any) -> dict[str, Any]:
    """Validate the final action of the turn and speak it."""
    action = AgentAction(kind=kind, speech=speech, **kw)
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
    if intent in (Intent.CANCEL, Intent.LIST):
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


# ---- nodes -----------------------------------------------------------------------------------
SLOT_QUESTIONS = {
    "specialty": "Which kind of doctor would you like to see: general medicine, "
    "dentistry, pediatrics or dermatology?",
    "date": "Which day would you like to come?",
    "time": "What time would suit you?",
    "patient_name": "May I have the patient's full name?",
    "patient_phone": "And a phone number to reach you?",
}
# Nodes reached after routing; each one ends the turn.
ACTION_NODES = (
    "booking",
    "book",
    "manage",
    "do_cancel",
    "decline_flow",
    "faq",
    "smalltalk",
    "handoff",
)
OUTAGE_APOLOGY = (
    "Sorry, I am having a technical problem. Please try again in a moment, "
    "or say staff and I will connect you with a person."
)
FLOW_ENDED: dict[str, Any] = {"flow": "none", "awaiting": "none"}


def _dump(slots: BookingSlots) -> dict[str, Any]:
    return slots.model_dump(mode="json")


def _safe(node: NodeFn) -> NodeFn:
    """A provider outage must degrade into an apology, never kill the voice session."""

    @functools.wraps(node)
    async def wrapper(self: Any, state: State) -> dict[str, Any]:
        try:
            return await node(self, state)
        except Exception:
            log.exception("node %s failed", node.__name__)
            return {"intent": Intent.OTHER.value, "action": _act("error", OUTAGE_APOLOGY)}

    return cast(NodeFn, wrapper)


class ConversationNodes:
    """The graph nodes. Each takes the state and returns the patch to apply to it."""

    def __init__(self, deps: Deps) -> None:
        self._deps = deps
        self._llm = deps.llm
        self._tools = deps.tools
        # Slot extraction started while the router is still deciding (see `route`).
        self._early: tuple[tuple[Any, ...], asyncio.Task[SlotUpdate]] | None = None

    # ---- routing
    @_safe
    async def route(self, state: State) -> dict[str, Any]:
        text = last_user_text(state)
        if not text.strip():
            return {"intent": Intent.OTHER.value, "action": {}}
        system = prompt(
            "router",
            flow=state.get("flow", "none"),
            awaiting=state.get("awaiting", "none") != "none",
        )
        self._start_early_extract(state)
        try:
            result = await structured_call(
                self._llm,
                "route",
                IntentResult,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript(state, 4)},
                ],
            )
            intent = result.intent
        except StructuredOutputError:
            intent = Intent.OTHER
        return {"intent": intent.value, "action": {}}

    def after_route(self, state: State) -> str:
        if not last_user_text(state).strip() or state.get("action", {}).get("kind") == "error":
            nxt = END
        else:
            nxt = decide(state, Intent(state["intent"]))
        if nxt not in ("booking", "manage"):
            self._drop_early_extract()
        return nxt

    # ---- slot extraction
    @staticmethod
    def _known(state: State, flow: str) -> dict[str, Any]:
        """What the extractor is told is already known, per flow."""
        if flow == "booking":
            slots = BookingSlots(**(state.get("slots") or {}))
            return {k: str(v) for k, v in slots.model_dump().items() if v is not None}
        return {"phone": state["phone"]} if state.get("phone") else {}

    @staticmethod
    def _early_key(state: State, known: dict[str, Any]) -> tuple[Any, ...]:
        return (len(state.get("messages", [])), last_user_text(state), sorted(known.items()))

    def _start_early_extract(self, state: State) -> None:
        """In the middle of a booking or cancel flow the next node almost always extracts slots,
        so start that call now, in parallel with the router, instead of after it. The result is
        used only if routing lands on that flow with the same known slots; otherwise it is dropped."""
        self._drop_early_extract()
        flow = state.get("flow", "none")
        if flow == "none" or state.get("awaiting", "none") != "none":
            return
        known = self._known(state, flow)
        task = asyncio.ensure_future(self._extract(state, known))
        self._early = (self._early_key(state, known), task)

    def _drop_early_extract(self) -> None:
        if self._early is not None:
            self._early[1].cancel()
            self._early = None

    async def _extract_slots(self, state: State, known: dict[str, Any]) -> SlotUpdate:
        early, self._early = self._early, None
        if early is not None:
            key, task = early
            if key == self._early_key(state, known):
                return await task
            task.cancel()
        return await self._extract(state, known)

    async def _extract(self, state: State, known: dict[str, Any]) -> SlotUpdate:
        today = self._deps.today()
        system = prompt(
            "extract_booking",
            today=today.isoformat(),
            weekday=today.strftime("%A"),
            calendar=calendar_text(today),
            specialties=", ".join(SPECIALTIES),
            known=known or "nothing yet",
        )
        try:
            update = await structured_call(
                self._llm,
                "extract",
                SlotUpdate,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript(state, 4)},
                ],
            )
        except StructuredOutputError:
            return SlotUpdate()
        update.date = resolve_weekday(last_user_text(state), today, update.date)
        return update

    # ---- booking flow
    def _sanitise(self, update: SlotUpdate) -> str:
        """Drop unusable specialty or past dates from the update; returns a note to speak."""
        note = ""
        if update.specialty:
            wanted = update.specialty.lower()
            match = next((s for s in SPECIALTIES if s in wanted or wanted in s), None)
            update.specialty = match
            if match is None:
                note = f"We offer {', '.join(SPECIALTIES[:-1])} and {SPECIALTIES[-1]}. "
        if update.date and update.date < self._deps.today():
            update.date = None
            note = "That date has already passed. "
        return note

    @_safe
    async def booking(self, state: State) -> dict[str, Any]:
        slots = BookingSlots(**(state.get("slots") or {}))
        known = self._known(state, "booking")
        update = await self._extract_slots(state, known)
        note = self._sanitise(update)
        slots = slots.merge(update)
        patch: dict[str, Any] = {"flow": "booking", "awaiting": "none"}

        if slots.specialty and slots.date and slots.time is None:
            return patch | await self._offer_times(slots, note)

        patch["slots"] = _dump(slots)
        if missing := slots.missing():
            patch["action"] = _act("ask_slot", note + SLOT_QUESTIONS[missing[0]])
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

    async def _offer_times(self, slots: BookingSlots, note: str) -> dict[str, Any]:
        """Specialty and date are known: offer free times, or ask for another day."""
        assert slots.specialty and slots.date
        day = slots.date
        try:
            # Ask for extra slots: several doctors can share a start time, and each time is said once.
            free = await self._tools.check_availability(
                slots.specialty, day.isoformat(), OFFERED_TIMES * 4
            )
            problem = "" if free else f"{note}There is nothing free that day. "
        except ToolError as err:
            free, problem = [], f"{err.message}. "
        if problem:
            slots = slots.model_copy(update={"date": None})
            return {
                "slots": _dump(slots),
                "action": _act("ask_slot", f"{problem}Which other day would suit you?"),
            }
        times = ", ".join(distinct_times(free))
        return {
            "slots": _dump(slots),
            "action": _act(
                "offer_slots",
                f"{note}On {say_date(day)} I have {times}. Which time do you prefer?",
            ),
        }

    @_safe
    async def book(self, state: State) -> dict[str, Any]:
        s = BookingSlots(**state["slots"])
        assert s.date and s.time and s.specialty and s.patient_name and s.patient_phone
        try:
            appt = await self._tools.book(
                specialty=s.specialty,
                date=s.date.isoformat(),
                start_time=s.time.strftime("%H:%M"),
                patient_name=s.patient_name,
                patient_phone=s.patient_phone,
                reason=s.reason or "",
            )
        except ToolError as err:
            return self._booking_failed(s, err)
        return {
            **FLOW_ENDED,
            "slots": {},
            "action": _act(
                "booked",
                f"Done. Your {say_appt(appt)} is booked. Is there anything else I can help with?",
                appointment_id=appt.appointment_id,
            ),
        }

    @staticmethod
    def _booking_failed(slots: BookingSlots, err: ToolError) -> dict[str, Any]:
        """A taken slot or a closed day only invalidates that slot; anything else ends the flow."""
        if err.code == "ALREADY_EXISTS":
            kept, ask_for = slots.model_copy(update={"time": None}), "time"
        elif err.code == "INVALID_ARGUMENT":
            kept, ask_for = slots.model_copy(update={"time": None, "date": None}), "day"
        else:
            return {
                **FLOW_ENDED,
                "flow": "none",
                "slots": {},
                "action": _act(
                    "error",
                    "I could not complete the booking right now. Would you like me to connect "
                    "you with a staff member?",
                ),
            }
        return {
            "awaiting": "none",
            "slots": _dump(kept),
            "action": _act(
                "ask_slot", f"Sorry, {_friendly(err)}. Please choose another {ask_for}."
            ),
        }

    # ---- cancel / list flow
    @_safe
    async def manage(self, state: State) -> dict[str, Any]:
        intent = Intent(state["intent"])
        if intent is Intent.CANCEL:
            flow = "cancel"
        elif intent is Intent.LIST:
            flow = "list"
        else:
            flow = state.get("flow", "cancel")
        known = self._known(state, flow)
        update = await self._extract_slots(state, known)
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
            appts = await self._tools.list_appointments(phone)
        except ToolError as err:
            patch["action"] = _act("error", f"{err.message}. Could you repeat the phone number?")
            patch["phone"] = None
            return patch

        if not appts:
            speech = "I don't see any upcoming appointments under that number."
        elif flow == "list":
            plural = "s" if len(appts) > 1 else ""
            listing = "; ".join(say_appt(a) for a in appts)
            speech = f"You have {len(appts)} appointment{plural}: {listing}."
        else:
            return self._propose_cancellation(patch, appts, update, phone)
        return {"flow": "none", "phone": None, "action": _act("appointments", speech)}

    @staticmethod
    def _propose_cancellation(
        patch: dict[str, Any], appts: list[AppointmentInfo], update: SlotUpdate, phone: str
    ) -> dict[str, Any]:
        chosen = appts
        if update.appointment_date:
            wanted = update.appointment_date.isoformat()
            chosen = [a for a in appts if a.date == wanted] or appts
        if len(chosen) > 1:
            listing = " or ".join(say_appt(a) for a in chosen)
            patch["action"] = _act(
                "ask_slot", f"You have several: {listing}. Which one should I cancel?"
            )
            return patch
        target = chosen[0]
        patch["awaiting"] = "cancel_confirm"
        patch["target"] = {"id": target.appointment_id, "phone": phone}
        patch["action"] = _act(
            "cancel_confirm",
            f"I found your {say_appt(target)}. Do you want me to cancel it?",
            appointment_id=target.appointment_id,
        )
        return patch

    @_safe
    async def do_cancel(self, state: State) -> dict[str, Any]:
        target = state["target"] or {}
        reset = {**FLOW_ENDED, "phone": None, "target": None}
        try:
            appt = await self._tools.cancel(target["id"], target["phone"])
        except ToolError as err:
            return {**reset, "action": _act("error", f"I could not cancel it: {err.message}.")}
        return {
            **reset,
            "action": _act(
                "cancelled",
                f"Your {say_appt(appt)} is cancelled. Anything else I can do for you?",
                appointment_id=appt.appointment_id,
            ),
        }

    @_safe
    async def decline_flow(self, state: State) -> dict[str, Any]:
        return {
            **FLOW_ENDED,
            "slots": {},
            "phone": None,
            "target": None,
            "action": _act(
                "decline_flow", "No problem, I have dropped that. Anything else I can help with?"
            ),
        }

    # ---- free-form replies (streamed token by token)
    async def _stream_reply(self, task: str, system: str, state: State) -> str:
        messages = [{"role": "system", "content": system}]
        for m in state.get("messages", [])[-self._deps.history :]:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            messages.append({"role": role, "content": str(m.content)})
        parts: list[str] = []
        async for token in self._llm.stream(task, messages):
            parts.append(token)
            _emit(token)
        return "".join(parts)

    @_safe
    async def faq(self, state: State) -> dict[str, Any]:
        hits = await self._deps.search(last_user_text(state))
        if not is_relevant(hits, self._deps.min_similarity):
            return {
                "action": _act(
                    "answer",
                    "I'm not sure about that one. I can connect you with a staff member if you like.",
                )
            }
        context = "\n\n".join(f"[{h.source}] {h.heading}: {h.content}" for h in hits)
        text = await self._stream_reply("answer", prompt("faq_answer", context=context), state)
        action = AgentAction(
            kind="answer",
            speech=text or "Sorry, I have no answer.",
            sources=sorted({h.source for h in hits}),
            intent=Intent.INFO,
        )
        return {"action": action.model_dump(mode="json")}

    @_safe
    async def smalltalk(self, state: State) -> dict[str, Any]:
        text = await self._stream_reply("smalltalk", prompt("smalltalk"), state)
        action = AgentAction(kind="greeting", speech=text or "Hello, how can I help?")
        return {"action": action.model_dump(mode="json")}

    @_safe
    async def handoff(self, state: State) -> dict[str, Any]:
        return {
            **FLOW_ENDED,
            "action": _act(
                "handoff",
                "Of course. I am passing you to a member of our staff now. "
                "If this is a medical emergency, please call the emergency number straight away.",
                handoff=True,
            ),
        }


def build_graph(deps: Deps) -> CompiledStateGraph:
    """Compile the conversation graph; state is checkpointed per thread (one thread per room)."""
    nodes = ConversationNodes(deps)
    graph = StateGraph(State)
    graph.add_node("route", nodes.route)
    for name in ACTION_NODES:
        graph.add_node(name, getattr(nodes, name))
        graph.add_edge(name, END)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", nodes.after_route, [*ACTION_NODES, END])
    return graph.compile(checkpointer=MemorySaver())
