import cv2
import mediapipe as mp
import pyautogui
import time
import math

# disable failsafe
pyautogui.FAILSAFE = False

# screen size
screen_w, screen_h = pyautogui.size()

# previous cursor position
prev_x, prev_y = 0, 0

# click timing control
last_click_time = 0
click_delay = 0.6

# camera active region limits
cam_x_min, cam_x_max = 0.1, 0.9
cam_y_min, cam_y_max = 0.1, 0.9


# mediapipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# mac camera setup
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


while True:

    success, img = cap.read()

    if not success:
        print("Camera not detected")
        break

    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            # index finger tip
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # thumb tip
            x2 = int(handLms.landmark[4].x * w)
            y2 = int(handLms.landmark[4].y * h)

            # draw points
            cv2.circle(img, (x, y), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0,0,255), cv2.FILLED)

            cv2.line(img, (x, y), (x2, y2), (255,0,0), 2)

            # normalized coords
            nx = handLms.landmark[8].x
            ny = handLms.landmark[8].y

            nx = max(cam_x_min, min(cam_x_max, nx))
            ny = max(cam_y_min, min(cam_y_max, ny))

            # convert to screen coords
            screen_x = (nx - cam_x_min) * screen_w / (cam_x_max - cam_x_min)
            screen_y = (ny - cam_y_min) * screen_h / (cam_y_max - cam_y_min)

            
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = screen_x, screen_y

            dx = screen_x - prev_x
            dy = screen_y - prev_y

            if abs(dx) < 2:
                dx = 0

            if abs(dy) < 2:
                dy = 0

            distance_move = math.hypot(dx, dy)

            if distance_move < 5:
                factor = 10
            elif distance_move < 20:
                factor = 7
            else:
                factor = 4

            curr_x = prev_x + dx / factor
            curr_y = prev_y + dy / factor

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # click detection
            distance = math.hypot(x2 - x, y2 - y)

            current_time = time.time()

            

            if distance < 50 and (current_time - last_click_time) > click_delay:

                pyautogui.click()

                last_click_time = current_time

                cv2.putText(
                    img,
                    "CLICK",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Advanced Hand Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)


cap.release()
cv2.destroyAllWindows()