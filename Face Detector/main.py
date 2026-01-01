import cv2
import os
from typing import Sequence

# Initializing and Configuring Camera
Camera: cv2.VideoCapture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not Camera.isOpened():
    print("Failed to open Camera!")
    exit(0)

Camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Camera.set(cv2.CAP_PROP_FPS, 30)

# Configuring Screen
cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

# ML Model
FaceCascade: cv2.CascadeClassifier = cv2.CascadeClassifier(os.path.join(os.path.dirname(cv2.__file__), "data/haarcascade_frontalface_default.xml"))

# Size Trackers
MinimumSize: list[float] = [1e9, 1e9]
AverageSize: list[float] = [0, 0]
MaximumSize: list[float] = [0, 0]
TotalSize: list[float] = [0, 0]
HDFrameCounter: int = 0

# Human Detection Variables
ThresholdSize: list[float] = [100, 100]
HumanDetected: bool = False
HumanFar: bool = False

# .........
def writeText(img: cv2.typing.MatLike, text: str, coordinate: cv2.typing.Point):
    cv2.putText(
        img,
        text,                     # text
        coordinate,               # bottom-left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, # font
        1,                        # font scale
        (0, 0, 255),              # color (BGR)
        2,                        # thickness
        cv2.LINE_AA               # line type
    )

def printRects(base: int) -> None:
    writeText(img, f"Min: {MinimumSize}", (0, base))
    writeText(img, f"Avg: [{AverageSize[0]:.2f}, {AverageSize[1]:.2f}]", (0, base + 50))
    writeText(img, f"Max: {MaximumSize}", (0, base + 100))

# Main Loop
while True:
    ret, img = Camera.read()
    if not ret:
        continue

    gray: cv2.typing.MatLike = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Faces: Sequence[cv2.typing.Rect] = FaceCascade.detectMultiScale(gray, 1.1, 5)

    if len(Faces) != 0:
        x, y, width, height = map(int, Faces[0])
        HDFrameCounter += 1

        if width < MinimumSize[0]:
            MinimumSize[0] = width
        elif width > MaximumSize[0]:
            MaximumSize[0] = width

        if height < MinimumSize[1]:
            MinimumSize[1] = height
        elif height > MaximumSize[1]:
            MaximumSize[1] = height

        TotalSize[0] += width
        TotalSize[1] += height

        AverageSize[0] = TotalSize[0] / HDFrameCounter
        AverageSize[1] = TotalSize[1] / HDFrameCounter

        if width > ThresholdSize[0] and height > ThresholdSize[1]:
            HumanDetected = True
            HumanFar = False
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5)
        else:
            HumanFar = True
    else:
        HumanDetected = False
        
    if HumanDetected:
        if HumanFar:
            writeText(img, "Human is far from Camera", (0, 50))
        else:
            writeText(img, "Human Detected", (0, 50))
    else:
        writeText(img, "Human not Detected", (0, 50))

    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break

# Exit Sequence
Camera.release()
cv2.destroyAllWindows()