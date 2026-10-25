"use strict";

const $ = (id) => document.getElementById(id);
const ui = {
  connect: $("connect"), mute: $("mute"), state: $("state"), banner: $("banner"), timer: $("timer"),
  transcript: $("transcript"), chat: $("chat"), text: $("text"), send: $("chat").querySelector("button"),
};

let room = null;
let maxCallSeconds = 0;
let callTimer = null;

// ---- small UI helpers ----------------------------------------------------------------------
function setState(state) {
  ui.state.textContent = state;
  ui.state.dataset.state = state;
}

function showBanner(message, kind = "error") {
  ui.banner.textContent = message;
  ui.banner.className = kind === "info" ? "banner info" : "banner";
  ui.banner.hidden = false;
}

function clearBanner() {
  ui.banner.hidden = true;
}

function setConnectButton(label, { busy = false, inCall = false } = {}) {
  ui.connect.textContent = label;
  ui.connect.disabled = busy;
  ui.connect.setAttribute("aria-busy", String(busy));
  ui.connect.classList.toggle("danger", inCall);
}

function setChatEnabled(enabled) {
  ui.text.disabled = ui.send.disabled = !enabled;
}

function setMuteLabel(muted) {
  ui.mute.textContent = muted ? "Unmute mic" : "Mute mic";
  ui.mute.setAttribute("aria-pressed", String(muted));
}

function formatClock(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function startCallTimer() {
  if (!maxCallSeconds) return;
  const startedAt = Date.now();
  const render = () => {
    ui.timer.textContent = `${formatClock((Date.now() - startedAt) / 1000)} / ${formatClock(maxCallSeconds)} (demo limit)`;
  };
  render();
  ui.timer.hidden = false;
  callTimer = setInterval(render, 1000);
}

function stopCallTimer() {
  clearInterval(callTimer);
  ui.timer.hidden = true;
}

function addLine(role, text, segmentId) {
  const box = ui.transcript;
  const stickToBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.querySelector(".empty")?.remove();
  let el = segmentId ? box.querySelector(`[data-seg="${CSS.escape(segmentId)}"]`) : null;
  if (!el) {
    el = document.createElement("p");
    el.className = role;
    if (segmentId) el.dataset.seg = segmentId;
    box.appendChild(el);
  }
  el.textContent = text;
  if (stickToBottom) box.scrollTop = box.scrollHeight;
}

function resetTranscript() {
  const empty = document.createElement("p");
  empty.className = "muted empty";
  empty.textContent = "The conversation will appear here.";
  ui.transcript.replaceChildren(empty);
}

// ---- server calls --------------------------------------------------------------------------
async function describeFailure(res) {
  const detail = await res.json().then((b) => b.detail).catch(() => null);
  if (typeof detail === "string" && (res.status === 429 || res.status === 503)) return detail;
  return `The server refused the call (error ${res.status}). Please try again.`;
}

async function fetchToken() {
  let res;
  try {
    res = await fetch("/api/token", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Caller" }),
    });
  } catch {
    throw new Error("Cannot reach the server. Check your connection and try again.");
  }
  if (!res.ok) throw new Error(await describeFailure(res));
  return res.json();
}

fetch("/api/config").then((r) => r.json()).then((c) => {
  $("clinic").textContent = c.clinic;
  $("version").textContent = c.version;
  maxCallSeconds = Number(c.max_call_seconds) || 0;
}).catch(() => {});

// ---- the call ------------------------------------------------------------------------------
function watchTranscript(lkRoom) {
  lkRoom.registerTextStreamHandler("lk.transcription", async (reader, info) => {
    const isAgent = info.identity !== lkRoom.localParticipant.identity;
    const segment = (reader.info.attributes || {})["lk.segment_id"] || reader.info.id;
    try {
      if (isAgent) {
        let text = "";
        for await (const chunk of reader) { text += chunk; addLine("agent", text, segment); }
      } else {
        addLine("user", await reader.readAll(), segment);
      }
    } catch {
      addLine("system", "Part of the transcript could not be received.");
    }
  });
}

function watchRoom(lkRoom) {
  const { RoomEvent, Track } = window.LivekitClient;
  lkRoom.on(RoomEvent.ParticipantAttributesChanged, (attrs, participant) => {
    if (participant.identity !== lkRoom.localParticipant.identity && attrs["lk.agent.state"]) {
      setState(attrs["lk.agent.state"]);
    }
  });
  lkRoom.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
    if (topic !== "turn_metrics") return;
    const metrics = JSON.parse(new TextDecoder().decode(payload));
    document.querySelectorAll("#latency [data-k]").forEach((cell) => {
      const value = metrics[cell.dataset.k];
      cell.textContent = value === undefined ? "-" : `${Number(value).toFixed(2)} s`;
    });
  });
  lkRoom.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) document.body.appendChild(track.attach());
  });
  lkRoom.on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((el) => el.remove()));
  lkRoom.on(RoomEvent.Disconnected, () => endCall());
}

function endCall() {
  room = null;
  stopCallTimer();
  setState("offline");
  setConnectButton("Start call");
  setChatEnabled(false);
  ui.mute.disabled = true;
  setMuteLabel(false);
}

async function enableMicrophone() {
  try {
    await room.localParticipant.setMicrophoneEnabled(true);
    ui.mute.disabled = false;
  } catch {
    showBanner("Microphone unavailable. You can type your messages instead.", "info");
    ui.text.focus();
  }
}

async function connectRoom(lkRoom, url, token) {
  try {
    await lkRoom.connect(url, token);
  } catch {
    throw new Error("Could not reach the call server. Please try again in a moment.");
  }
}

async function startCall() {
  clearBanner();
  resetTranscript();
  setConnectButton("Connecting...", { busy: true });
  setState("connecting");
  try {
    const { url, token } = await fetchToken();
    room = new window.LivekitClient.Room();
    watchTranscript(room);
    watchRoom(room);
    await connectRoom(room, url, token);
    await room.startAudio(); // inside the click gesture, so browsers allow playback
    setConnectButton("End call", { inCall: true });
    setChatEnabled(true);
    startCallTimer();
    await enableMicrophone(); // enables the mute button once the microphone is granted
  } catch (err) {
    showBanner(err.message || "Could not start the call.");
    await room?.disconnect().catch(() => {});
    endCall();
  }
}

ui.connect.onclick = async () => {
  if (room) await room.disconnect();
  else await startCall();
};

ui.mute.onclick = async () => {
  const muted = room.localParticipant.isMicrophoneEnabled;
  await room.localParticipant.setMicrophoneEnabled(!muted);
  setMuteLabel(muted);
};

ui.chat.onsubmit = async (event) => {
  event.preventDefault();
  const text = ui.text.value.trim();
  if (!text || !room) return;
  ui.text.value = "";
  addLine("user", text);
  try {
    await room.localParticipant.sendText(text, { topic: "lk.chat" });
  } catch {
    showBanner("Your message could not be sent. Please try again.");
  }
  ui.text.focus();
};

if (!window.LivekitClient) {
  ui.connect.disabled = true;
  showBanner("The call library could not be loaded. Check your connection and reload the page.");
}
