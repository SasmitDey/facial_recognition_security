import threading
import traceback
import time
import av
import cv2
import numpy as np
import streamlit as st
from insightface.app import PersonAnalysis
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Facial Recognition and Suspect Tracking", layout="wide")


@st.cache_resource
def get_model():
    m = PersonAnalysis(name="cheetah_s", root="./models")
    # insightface-style apps need prepare() before get(), otherwise detection
    # can silently return nothing. Harmless if your build doesn't have it.
    if hasattr(m, "prepare"):
        try:
            m.prepare(ctx_id=0, det_size=(640, 640))
        except TypeError:
            try:
                m.prepare(ctx_id=0)
            except Exception:
                pass
    return m


@st.cache_resource
def get_model_lock():
    return threading.Lock()


model = get_model()
model_lock = get_model_lock()


def pick(obj, *names, default=None):
    """Return the first attribute (or dict key) that exists and isn't None."""
    if obj is None:
        return default
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return default


def find_bbox(obj):
    """Look for a bbox on the object itself, or on a nested face/detection object."""
    if obj is None:
        return None
    bbox = pick(obj, "bbox", "box", "rect", "xyxy")
    if bbox is not None:
        return bbox
    inner = pick(obj, "face", "detection", "det", "query")
    if inner is not None and inner is not obj:
        return pick(inner, "bbox", "box", "rect", "xyxy")
    return None


NAME_FIELDS = ("matched_name", "name", "person_name", "identity", "label",
               "matched_id", "person_id", "person", "id")
UNKNOWN_VALUES = {"", "unknown", "none", "null", "-1"}


def get_name(match):
    """Name from a match object, or None if it's a non-match."""
    if match is None:
        return None
    # Explicit "no match" flags win over any name field
    for flag in ("is_match", "matched", "is_known", "known", "found"):
        v = pick(match, flag)
        if v is not None and not bool(v):
            return None
    v = pick(match, *NAME_FIELDS)
    # name could itself be an object (e.g. a Person record)
    if v is not None and not isinstance(v, (str, int, np.integer)):
        v = pick(v, "name", "label", "id", default=v)
    if v is None or str(v).strip().lower() in UNKNOWN_VALUES:
        return None
    return str(v)


def unwrap_matches(matches):
    """match() may return one result object holding the list instead of a list."""
    if matches is None:
        return []
    if isinstance(matches, (list, tuple)):
        return list(matches)
    inner = pick(matches, "matches", "results", "items", "persons", "faces")
    if inner is not None and not isinstance(inner, (str, bytes)):
        return list(inner)
    try:
        return list(matches)
    except TypeError:
        return [matches]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pair_faces_and_matches(faces, matches, w, h):
    """Attach each match to a face: by bbox overlap, by a face reference, or by index."""
    pairs = {i: None for i in range(len(faces))}
    face_boxes = [to_pixel_box(find_bbox(f), w, h) for f in faces]
    leftovers = []
    for j, m in enumerate(matches):
        ref = pick(m, "face", "query")
        idx = pick(m, "face_index", "face_idx", "index", "query_index")
        target = None
        if ref is not None:
            target = next((i for i, f in enumerate(faces) if f is ref), None)
        if target is None and isinstance(idx, (int, np.integer)) and 0 <= idx < len(faces):
            target = int(idx)
        if target is None:
            mb = to_pixel_box(find_bbox(m), w, h)
            if mb is not None:
                scores = [iou(mb, fb) if fb else 0 for fb in face_boxes]
                if scores and max(scores) > 0.3:
                    target = int(np.argmax(scores))
        if target is not None and pairs[target] is None:
            pairs[target] = m
        else:
            leftovers.append(m)
    # 3. same count -> assume same order
    if leftovers and len(matches) == len(faces):
        for i in pairs:
            if pairs[i] is None:
                pairs[i] = matches[i]
    return [(faces[i], pairs[i]) for i in range(len(faces))]


def to_pixel_box(bbox, w, h):
    """Turn whatever bbox format we got into clipped integer (x1, y1, x2, y2)."""
    try:
        vals = np.asarray(bbox, dtype=float).reshape(-1)[:4]
    except Exception:
        return None
    if vals.size < 4 or not np.all(np.isfinite(vals)):
        return None
    x1, y1, a, b = vals
    if max(abs(x1), abs(y1), abs(a), abs(b)) <= 1.5:
        x1, a = x1 * w, a * w
        y1, b = y1 * h, b * h
    if a <= x1 or b <= y1:
        a, b = x1 + a, y1 + b
    x1, x2 = int(max(0, min(w - 1, x1))), int(max(0, min(w - 1, a)))
    y1, y2 = int(max(0, min(h - 1, y1))), int(max(0, min(h - 1, b)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self._pending = None
        self._results = []       
        self.last_error = None
        self.debug_info = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _infer_loop(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                img, self._pending = self._pending, None
            if img is None:
                continue
            try:
                with model_lock:
                    faces = model.get(img)
                    matches = model.match(faces)
                    model.update(matches)

                raw_matches = matches
                faces = list(faces) if faces is not None else []
                matches = unwrap_matches(matches)

                h, w = img.shape[:2]
                if faces:
                    results = pair_faces_and_matches(faces, matches, w, h)
                else:
                    results = [(None, m) for m in matches if find_bbox(m) is not None]

                first_bbox = None
                for f, m in results:
                    b = find_bbox(m) or find_bbox(f)
                    if b is not None:
                        first_bbox = np.asarray(b).tolist()
                        break

                with self._lock:
                    self._results = results
                    self.last_error = None
                    self.debug_info = {
                        "frame_shape": list(img.shape),
                        "faces_found": len(faces),
                        "matches_found": len(matches),
                        "match_return_type": type(raw_matches).__name__,
                        "first_match_repr": repr(matches[0])[:400] if matches else None,
                        "names": [get_name(m) for _, m in results],
                        "registered_count": pick(model, "num_registered", "size", "count",
                                                 default=None),
                        "first_bbox_raw": first_bbox,
                        "face_type": type(faces[0]).__name__ if faces else None,
                        "face_fields": (list(faces[0].keys()) if isinstance(faces[0], dict)
                                        else [a for a in dir(faces[0]) if not a.startswith("_")]) if faces else None,
                        "match_type": type(matches[0]).__name__ if matches else None,
                        "match_fields": (list(matches[0].keys()) if isinstance(matches[0], dict)
                                         else [a for a in dir(matches[0]) if not a.startswith("_")]) if matches else None,
                    }
            except Exception:
                with self._lock:
                    self.last_error = traceback.format_exc()
                    self._results = []

    def on_ended(self):
        self._stop.set()
        self._wake.set()

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_status(self):
        with self._lock:
            return self.last_error, dict(self.debug_info), [
                self._describe(f, m) for f, m in self._results
            ]

    @staticmethod
    def _describe(face, match):
        name = get_name(match)
        score = pick(match, "score", "similarity", "confidence", "sim", "distance")
        det = pick(face if face is not None else match, "det_score", "detection_score")
        return {"name": name, "match_score": score, "det_score": det}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        with self._lock:
            self._latest_frame = img.copy()
            self._pending = img.copy()
            results = list(self._results)
        self._wake.set()

        for face, match in results:
            box = to_pixel_box(find_bbox(face) if face is not None else find_bbox(match), w, h)
            if box is None and match is not None:
                box = to_pixel_box(find_bbox(match), w, h)
            if box is None:
                continue

            info = self._describe(face, match)
            known = info["name"] is not None
            color = (0, 0, 255) if known else (0, 255, 0)
            label = f"{info['name']}" if known else "UNKNOWN"
            try:
                if info["match_score"] is not None:
                    label += f" {float(info['match_score']):.2f}"
                elif info["det_score"] is not None:
                    label += f" det {float(info['det_score']):.2f}"
            except (TypeError, ValueError):
                pass

            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="checkpoint-surveillance",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    media_stream_constraints={
        "audio": False,
        "video": {"width": 640, "height": 480, "frameRate": {"ideal": 20}},
    },
    media_toggle_controls=False,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# ==========================================
# SIDEBAR REGISTRATION FORM
# ==========================================
st.sidebar.title("Register Face", help="Register current frame face into RAM")

with st.sidebar.form("register_form", clear_on_submit=True):
    name = st.text_input("Enter name of subject", key="subject_name")
    submit = st.form_submit_button("Register")

    if submit:
        frame = ctx.video_processor.get_latest_frame() if ctx.video_processor else None
        if not name:
            st.error("Please enter a valid name.")
        elif frame is None:
            st.warning("No active video feed detected.")
        else:
            with model_lock:
                result = model.register(name, frame)
            ok = result if isinstance(result, bool) else pick(result, "accepted", "success", "ok", default=1)
            if ok and (not isinstance(ok, (int, float)) or ok > 0):
                st.success(f"Registered '{name}' in RAM.")
            else:
                st.error("Registration failed: No clear face detected in frame.")

status_box = st.empty()
while ctx.state.playing and ctx.video_processor:
    err, dbg, people = ctx.video_processor.get_status()
    with status_box.container():
        if err:
            st.error("Inference error:")
            st.code(err)
        else:
            st.write(dbg)
            if people:
                st.table(people)
            else:
                st.info("No faces detected yet.")
    time.sleep(0.5)