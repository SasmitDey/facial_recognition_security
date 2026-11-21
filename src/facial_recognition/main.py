import cv2
from pathlib import Path
# from insightface.app import FaceAnalysis
from insightface.app import PersonAnalysis
from insightface.data import get_image

# app = FaceAnalysis(root="./models")

# app.prepare()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Build the exact paths (change "data" to ".data" if your folder actually has a dot)
ref_path = PROJECT_ROOT / "data" / "watchlist" / "reference.jpg"
cap_path = PROJECT_ROOT / "data" / "capture" / "captured.jpg"

cap_path.parent.mkdir(parents=True, exist_ok=True)

def take_picture():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(cap_path), frame)
        break

    cap.release()
    return frame
 

# Load the images (convert Path to string for OpenCV)
reference_img = cv2.imread(str(ref_path))
# captured_img = cv2.imread(str(cap_path))

# img = app.get(captured_img)
# cv2.imwrite("img.jpg", app.draw_on(captured_img, img))


#face matching


with PersonAnalysis(name="cheetah_s", root="./models") as app:
    app.register("sasmit", reference_img)
    captured_img = take_picture()
    matches = app.match(app.get(captured_img))
    try:
        matched = matches[0]
        print(f"Matched by: {matched.matched_by}, Similarity: {matched.similarity:.2f}, Person_id: {matched.person_id}")
    except Exception as e:
        print("no match found")