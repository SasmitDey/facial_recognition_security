# Facial Recognition and Suspect Tracking

A real-time face recognition app built with Streamlit. It takes the live webcam feed in the browser, detects faces, compares them against people registered during the session, and draws a labelled bounding box around each face.

- **Red box + name**: the face matches a registered person
- **Green box + `UNKNOWN`**: a face was detected but not recognised

Registered faces live in memory (RAM) only. Nothing is written to disk, and the list is cleared when the app restarts.

---

## Features

- Live webcam streaming in the browser over WebRTC (`streamlit-webrtc`)
- Face detection and recognition with the `PersonAnalysis` model (`cheetah_s`)
- One-click registration: type a name and capture the face from the current frame
- Inference runs in a background thread, so the video stays smooth while the model works
- Handles different bounding-box formats (corner coordinates, `x, y, w, h`, or normalised 0–1 values)
- Pairs detected faces with recognition results by face reference, index or box overlap
- Live debug panel showing detection counts, raw match output and per-face results

---

## Requirements

- Python 3.9 or newer
- A webcam, and a browser that allows camera access (Chrome, Edge or Firefox)
- An `insightface` build that provides `insightface.app.PersonAnalysis`, plus the `cheetah_s` model files

Python packages:

```
streamlit
streamlit-webrtc
opencv-python
av
numpy
insightface   # build that includes PersonAnalysis
```

> **Note:** The stock `insightface` package on PyPI provides `FaceAnalysis` (e.g. `buffalo_l`), not `PersonAnalysis`. This project needs the build or fork that includes `PersonAnalysis` and the `cheetah_s` model pack.

---

## Project structure

```
.
├── app.py              # Streamlit app (streaming, inference, registration, UI)
├── README.md
└── models/
    └── cheetah_s/      # model files loaded by PersonAnalysis
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install streamlit streamlit-webrtc opencv-python av numpy
pip install <your insightface build with PersonAnalysis>

# 3. Put the model files in ./models/cheetah_s/
```

---

## Running the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), click **START**, and allow camera access.

> Browsers only allow camera access on `localhost` or over HTTPS. To open the app from another device on your network, serve it over HTTPS (for example with a reverse proxy or a tunnelling tool).

---

## Using the app

1. **Start the feed.** Click **START** under the video. Faces should get green `UNKNOWN` boxes.
2. **Register a person.** Face the camera with good lighting, enter a name in the **Register Face** sidebar form, and click **Register**. The sidebar shows whether registration worked and what `register()` returned.
3. **Recognition.** From then on, that person's face gets a red box with their name and match score.
4. **Monitor.** The panel under the video shows the debug info and a table of detected faces with name, match score and detection score.

---

## How it works

```
Browser webcam
     │  (WebRTC)
     ▼
VideoProcessor.recv()  ── stores newest frame ──►  background inference thread
     │                                                 │
     │                                     model.get(img)      → detect faces
     │                                     model.match(faces)  → compare to registered
     │                                     model.update(...)   → update tracking state
     │                                                 │
     │◄──────── latest (face, match) pairs ────────────┘
     ▼
draw boxes + labels  →  frame returned to the browser
```

Main pieces of `app.py`:

| Component | Purpose |
|---|---|
| `get_model()` | Loads `PersonAnalysis` once (cached) and calls `prepare()` if it exists |
| `model_lock` | Shared lock so registration and inference never use the model at the same time |
| `VideoProcessor.recv()` | Runs for every frame: queues the frame for inference and draws the most recent results |
| `VideoProcessor._infer_loop()` | Background thread that runs detection and matching on the newest frame only, skipping stale ones |
| `pair_faces_and_matches()` | Attaches each match to the right face |
| `get_name()` | Gets the person's name from a match and treats "unknown" values or no-match flags as no match |
| `to_pixel_box()` | Converts any bbox format into clipped pixel coordinates |
| Sidebar form | Registers the face in the current frame under a name |

---

## Debug panel

| Field | Meaning |
|---|---|
| `frame_shape` | Size of the frames being processed |
| `faces_found` | Faces detected in the last processed frame |
| `matches_found` | Results returned by `match()` |
| `match_return_type` | Type `match()` returned (list or a wrapper object) |
| `first_match_repr` | Raw contents of the first match (shows field names and scores) |
| `names` | Name read for each face (`null` means unknown) |
| `first_bbox_raw` | Raw bounding box of the first face, before conversion |
| `face_fields` / `match_fields` | Fields available on the face and match objects |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| No boxes at all, `faces_found` is 0 | Detection is failing. Check the model files in `./models/cheetah_s` and that `prepare()` ran. Improve lighting. |
| `faces_found` > 0 but no boxes | The bbox is stored under an unexpected field name. Check `face_fields` and add the name to `find_bbox()`. |
| Boxes appear but always `UNKNOWN` after registering | Check `first_match_repr`. If your name is there with a low score, re-register from a clear, front-facing frame. If it's stored under a different field, add it to `NAME_FIELDS`. |
| Registration says it failed | No clear face in the captured frame. Face the camera directly and try again. |
| "Inference error" panel | Read the traceback shown. Usually the model isn't loaded correctly or the API differs from what the code expects. |
| Camera doesn't start | Allow camera permission, and use `localhost` or HTTPS. |
| Video connects locally but not over a network | WebRTC may need a TURN server in addition to the STUN server in `rtc_configuration`. |

---

## Limitations

- Registered faces are kept in RAM only and are lost when the app restarts.
- One reference image per registration. Registering the same person in several poses and lighting conditions improves accuracy.
- Accuracy depends on lighting, camera quality, face angle and the model's match threshold.
- Built for a single viewer and camera at a time.

---

## Responsible use

Facial recognition handles biometric data. Use this project only for learning, demos and testing, with people who have given informed consent to be registered and recognised. Don't use it to identify or track people without their knowledge, and check the privacy and data-protection laws that apply where you run it (for example India's Digital Personal Data Protection Act, 2023).
