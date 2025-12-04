#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import os
import math
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import smtplib
import datetime
import pyautogui
import keyboard
import serial, serial.tools.list_ports

try:
    arduino = serial.Serial('COM9', 9600, timeout=1)
    print("Serial connected!")
except serial.SerialException as e:
    print("Error opening serial port:", e)
    exit()


# Timers
EYES_CLOSED_START = None
BUZZER_TRIGGER_TIME = 4  # seconds

# Optional internet-dependent imports
try:
    import pywhatkit
    PYWHATKIT_AVAILABLE = True
except Exception:
    PYWHATKIT_AVAILABLE = False
    print("[WARNING] pywhatkit not available - WhatsApp alerts will be disabled")

try:
    import geocoder
    GEOCODER_AVAILABLE = True
except Exception:
    GEOCODER_AVAILABLE = False
    print("[WARNING] geocoder not available - location tracking will be disabled")

def get_device_location():
    try:
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            lat, lng = g.latlng
            return f"https://www.google.com/maps?q={lat},{lng}"
        return "Location unavailable"
    except Exception:
        return "Location error"

ALERT_SENT = False
EMERGENCY_THRESHOLD = 10  # WhatsApp after 10 sec

def send_whatsapp_alert():
    global ALERT_SENT

    if not PYWHATKIT_AVAILABLE:
        print("[WARNING] pywhatkit not available - skipping WhatsApp alerts")
        return

    location = get_device_location()
    msg = (
        "🚨 EMERGENCY ALERT 🚨\n"
        "The driver has been drowsy for more than 10 seconds.\n"
        "Immediate help is required!\n\n"
        f"📍 Driver Location:\n{location}"
    )

    contacts = [
        "+918999318374",
        "+918208760597",
        "+918830649200",
    ]

    for number in contacts:
        try:
            print(f"[INFO] Sending WhatsApp alert to {number}...")
            # Use instant send to avoid scheduling/timeout issues
            pywhatkit.sendwhatmsg_instantly(
                phone_no=number,
                message=msg,
                wait_time=20,   # seconds to allow WhatsApp Web to load
                tab_close=False
            )

            time.sleep(8)
            pyautogui.click(400, 400)
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(1)
            pyautogui.press("enter")

            print(f"[SUCCESS] WhatsApp alert sent to {number}!")
        except Exception as e:
            print(f"[ERROR] Failed to send alert to {number}: {e}")

    ALERT_SENT = True


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
base_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(base_dir, 'dlib_shape_predictor', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)

print("[INFO] initializing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(1.0)

frame_width = 1280
frame_height = 720

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", frame_width, frame_height)

image_points = np.array([
    (359, 391),
    (399, 561),
    (337, 297),
    (513, 301),
    (345, 465),
    (453, 469)
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
COUNTER = 0
HEAD_TILT_THRESH = 25

(mStart, mEnd) = (49, 68)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = imutils.resize(frame, width=frame_width, height=frame_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.require(gray, dtype=np.uint8, requirements=['C'])
    size = gray.shape

    rects = detector(gray, 0)
    eyes_status_display = None
    eyes_color_display = (0, 255, 0)

    face_count_text = "Faces: {}".format(len(rects))
    cv2.putText(frame, face_count_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if len(rects) == 0:
        cv2.putText(frame, "No face detected", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        eyes_status = "Open"
        eyes_color = (0, 255, 0)

        # ==========================================================
        # 🔥 ONLY THIS BLOCK WAS MODIFIED (minimal change)
        # ==========================================================
        if ear < EYE_AR_THRESH:
            eyes_status = "Closed"
            eyes_color = (0, 0, 255)

            if EYES_CLOSED_START is None:
                EYES_CLOSED_START = time.time()

            closed_duration = time.time() - EYES_CLOSED_START

            cv2.putText(frame, f"Closed for: {int(closed_duration)} sec",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

            # Buzzer after 4s
            if closed_duration >= BUZZER_TRIGGER_TIME:
                print(">>> Sending 1 to Arduino (buzzer ON)")
                try:
                    arduino.write(b'1')
                except serial.SerialException as e:
                    print(f"[WARNING] Failed to write to serial: {e}")

            # WhatsApp after 10 sec
            if closed_duration >= EMERGENCY_THRESHOLD and not ALERT_SENT:
                send_whatsapp_alert()

        else:
            EYES_CLOSED_START = None
            closed_duration = 0
            ALERT_SENT = False  # FIXED: reset only when eyes open

            print(">>> Sending 0 to Arduino (buzzer OFF)")
            try:
                arduino.write(b'0')
            except serial.SerialException as e:
                print(f"[WARNING] Failed to write to serial: {e}")
        # ==========================================================

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (170, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "Eyes: {}".format(eyes_status),
                    (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eyes_color, 2)

        eyes_status_display = eyes_status
        eyes_color_display = eyes_color

        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')

            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point,
         end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree is not None:
            tilt_val = float(head_tilt_degree[0])
            cv2.putText(frame, 'Head Tilt: {:.1f}°'.format(tilt_val), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    h, w = frame.shape[:2]
    banner_text = None
    banner_color = eyes_color_display

    if len(rects) == 0:
        banner_text = "NO FACE"
        banner_color = (0, 0, 255)
    else:
        if eyes_status_display:
            banner_text = "EYES CLOSED" if eyes_status_display == "Closed" else "EYES OPEN"

    if banner_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(banner_text, font, scale, thickness)
        x = (w - text_w) // 2
        y = 100
        cv2.putText(frame, banner_text, (x+2, y+2), font, scale, (0, 0, 0), thickness+2)
        cv2.putText(frame, banner_text, (x, y), font, scale, banner_color, thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
try:
    cap.release()
except Exception:
    pass