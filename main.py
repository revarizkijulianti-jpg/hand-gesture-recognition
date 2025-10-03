import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
pygame.mixer.init()
import os

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture = ""
last_gesture = ""   # Supaya suara tidak berulang tiap frame

def y(landmarks, id): return landmarks[id].y
def x(landmarks, id): return landmarks[id].x

# === Mapping gesture ke teks suara ===
gesture_speech = {
    "Halo Semua": "Halo Semua",
    "Perkenalkan": "Perkenalkan",
    "Nama Saya Reva": "Nama saya Reva",
    "Salam Kenal": "Salam Kenal",
    "Fist": "Fist",
    "Terimakasih": "Terimakasih",
    "Pinky Up": "Pinky Up"
}

# === fungsi play audio pakai pygame ===
def play_audio(filename):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print("Gagal putar suara:", e)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik tangan
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

            lm = hand_landmarks.landmark

            # === Halo Semua (semua jari ke atas) ===
            all_fingers_up = all(
                y(lm, tip) < y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_fingers_up:
                gesture = "Halo Semua"

            # === Perkenalkan (telunjuk naik, lain turun) ===
            thumb_middle_close = (
                abs(x(lm, mp_hands.HandLandmark.THUMB_TIP) - x(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)) < 0.05 and
                abs(y(lm, mp_hands.HandLandmark.THUMB_TIP) - y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)) < 0.05
            )
            index_up = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            if thumb_middle_close and index_up:
                gesture = "Perkenalkan"

            # === Nama Saya Reva (jempol, telunjuk, kelingking naik) ===
            saya_reva = (
                y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP) and
                y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.PINKY_TIP) < y(lm, mp_hands.HandLandmark.PINKY_PIP) and
                y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) > y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            )
            if saya_reva:
                gesture = "Nama Saya Reva"

           # === Salam Kenal (Telunjuk,Tengah naik, lainnya turun) 
            index_up = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_up = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            ring_down = y(lm, mp_hands.HandLandmark.RING_FINGER_TIP) > y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            pinky_down = y(lm, mp_hands.HandLandmark.PINKY_TIP) > y(lm, mp_hands.HandLandmark.PINKY_PIP)
            thumb_down = y(lm, mp_hands.HandLandmark.THUMB_TIP) > y(lm, mp_hands.HandLandmark.THUMB_IP)

            if index_up and middle_up and ring_down and pinky_down and thumb_down:
                gesture = "Salam Kenal"
            # === Fist (semua jari ditekuk) ===
            all_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if all_folded:
                gesture = "Fist"

            # === Terimakasih (jempol naik, lain turun) ===
            thumb_up = y(lm, mp_hands.HandLandmark.THUMB_TIP) < y(lm, mp_hands.HandLandmark.THUMB_IP)
            others_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            if thumb_up and others_folded and not all_folded:
                gesture = "Terimakasih"

            # === Pinky Up (kelingking naik, lain turun) ===
            pinky_up = y(lm, mp_hands.HandLandmark.PINKY_TIP) < y(lm, mp_hands.HandLandmark.PINKY_PIP)
            other_fingers_folded = all(
                y(lm, tip) > y(lm, pip) for tip, pip in [
                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                ]
            )
            if pinky_up and other_fingers_folded:
                gesture = "Pinky Up"

    # === tampilkan teks di layar ===
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # === mainkan suara kalau gesture baru ===
        if gesture != last_gesture and gesture in gesture_speech:
            filename = f"{gesture}.mp3"
            if not os.path.exists(filename):
                # bikin file mp3 otomatis
                tts = gTTS(text=gesture_speech[gesture], lang='id')
                tts.save(filename)
            play_audio(filename)
            last_gesture = gesture

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
