const $ = (id) => document.getElementById(id);
const { Room, RoomEvent } = LivekitClient;
let room = null;

function setState(s) {
  $("state").textContent = s;
  $("state").dataset.state = s;
}

function addLine(role, text, segmentId) {
  const box = $("transcript");
  box.querySelector(".empty")?.remove();
  let el = segmentId ? box.querySelector(`[data-seg="${CSS.escape(segmentId)}"]`) : null;
  if (!el) {
    el = document.createElement("p");
    el.className = role;
    if (segmentId) el.dataset.seg = segmentId;
    box.appendChild(el);
  }
  el.textContent = text;
  box.scrollTop = box.scrollHeight;
  return el;
}

fetch("/api/config").then((r) => r.json()).then((c) => {
  $("clinic").textContent = `${c.clinic} - ${c.note}`;
  $("version").textContent = c.version;
});

$("connect").onclick = async () => {
  if (room) { await room.disconnect(); return; }
  $("connect").disabled = true;
  try {
    const res = await fetch("/api/token", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Caller" }),
    });
    if (!res.ok) throw new Error(`token request failed (${res.status})`);
    const { url, token } = await res.json();
    room = new Room();

    room.registerTextStreamHandler("lk.transcription", async (reader, info) => {
      const attrs = reader.info.attributes || {};
      const isAgent = info.identity !== room.localParticipant.identity;
      const seg = attrs["lk.segment_id"] || reader.info.id;
      if (isAgent) {
        let text = "";
        for await (const chunk of reader) { text += chunk; addLine("agent", text, seg); }
      } else {
        addLine("user", await reader.readAll(), seg);
      }
    });
    room.on(RoomEvent.ParticipantAttributesChanged, (attrs, p) => {
      if (p.identity !== room.localParticipant.identity && attrs["lk.agent.state"]) {
        setState(attrs["lk.agent.state"]);
      }
    });
    room.on(RoomEvent.DataReceived, (payload, p, kind, topic) => {
      if (topic !== "turn_metrics") return;
      const m = JSON.parse(new TextDecoder().decode(payload));
      document.querySelectorAll("#latency [data-k]").forEach((td) => {
        const v = m[td.dataset.k];
        td.textContent = v === undefined ? "-" : `${Number(v).toFixed(2)} s`;
      });
    });
    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === "audio") document.body.appendChild(track.attach());
    });
    room.on(RoomEvent.Disconnected, () => {
      room = null; setState("offline");
      $("connect").textContent = "Start call";
      ["mute", "text"].forEach((i) => ($(i).disabled = true));
      $("chat").querySelector("button").disabled = true;
    });

    await room.connect(url, token);
    setState("connecting");
    $("connect").textContent = "End call";
    $("chat").querySelector("button").disabled = false;
    $("text").disabled = false;
    try {
      await room.localParticipant.setMicrophoneEnabled(true);
      $("mute").disabled = false;
    } catch (e) {
      addLine("agent", "Microphone unavailable. You can type your messages instead.");
    }
  } catch (err) {
    addLine("agent", `Could not start the call: ${err.message}`);
    room = null;
  } finally {
    $("connect").disabled = false;
  }
};

$("mute").onclick = async () => {
  const on = room.localParticipant.isMicrophoneEnabled;
  await room.localParticipant.setMicrophoneEnabled(!on);
  $("mute").textContent = on ? "Unmute mic" : "Mute mic";
};

$("chat").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text || !room) return;
  $("text").value = "";
  addLine("user", text);
  await room.localParticipant.sendText(text, { topic: "lk.chat" });
};
