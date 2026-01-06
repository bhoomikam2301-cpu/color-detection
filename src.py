import cv2
import numpy as np

# Use DirectShow backend (Windows fix)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

# Kernel for noise removal
kernel = np.ones((5, 5), "uint8")

# Color ranges dictionary (HSV)
colors = {
    "Red": {
        "lower": np.array([136, 87, 111]),
        "upper": np.array([180, 255, 255]),
        "bgr": (0, 0, 255)
    },
    "Green": {
        "lower": np.array([25, 52, 72]),
        "upper": np.array([102, 255, 255]),
        "bgr": (0, 255, 0)
    },
    "Blue": {
        "lower": np.array([94, 80, 2]),
        "upper": np.array([120, 255, 255]),
        "bgr": (255, 0, 0)
    },
    "Yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255]),
        "bgr": (0, 255, 255)
    },
    "Orange": {
        "lower": np.array([10, 100, 20]),
        "upper": np.array([25, 255, 255]),
        "bgr": (0, 165, 255)
    },
    "Purple": {
        "lower": np.array([125, 50, 50]),
        "upper": np.array([150, 255, 255]),
        "bgr": (255, 0, 255)
    }
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through all colors
    for color_name, color_info in colors.items():
        lower = color_info["lower"]
        upper = color_info["upper"]
        bgr = color_info["bgr"]

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

    cv2.imshow("Optimized Multi Color Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
