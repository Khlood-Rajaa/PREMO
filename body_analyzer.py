import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO

class ContinuousBodyLanguageAnalyzer:
    def __init__(self, max_history=15, mobile_model=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh()
        self.mobile_model = mobile_model  
        self.analysis_queue = deque(maxlen=max_history)
        
        # Stats for summary
        self.eye_contact_count = 0
        self.total_frames = 0
        self.crossed_count = 0
        self.pocket_count = 0
        self.mobile_count = 0
        self.moving_count = 0
        
        # Tracking consecutive behaviors
        self.consecutive_no_eye_contact = 0
        self.max_consecutive_no_eye_contact = 0
        self.consecutive_crossed = 0
        self.max_consecutive_crossed = 0
        self.consecutive_pocket = 0
        self.max_consecutive_pocket = 0

    # ----------------- Eye Contact -----------------
    def get_eye_contact(self, face_landmarks):
        left_eye_corners = [33, 133]
        right_eye_corners = [362, 263]
        landmarks = face_landmarks.landmark

        left_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_corners], axis=0)
        right_eye = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_corners], axis=0)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if not (0.03 < eye_distance < 0.15):
            return False

        nose_tip, left_face, right_face = landmarks[1], landmarks[234], landmarks[454]
        face_center_x = (left_face.x + right_face.x) / 2
        nose_offset_x = abs(nose_tip.x - face_center_x)
        face_width = abs(left_face.x - right_face.x)
        nose_offset_normalized = nose_offset_x / face_width if face_width > 0 else 1
        return nose_offset_normalized < 0.25

    # ----------------- Hand in Pocket -----------------
    def detect_hand_in_pocket(self, landmarks):
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP = 15, 16, 23, 24
        lw, rw, lh, rh = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST], landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
        lw_y_diff, rw_y_diff = abs(lw.y - lh.y), abs(rw.y - rh.y)
        lw_x_diff, rw_x_diff = abs(lw.x - lh.x), abs(rw.x - rh.x)
        left_in_pocket = lw_y_diff < 0.1 and lw_x_diff < 0.08
        right_in_pocket = rw_y_diff < 0.1 and rw_x_diff < 0.08
        return left_in_pocket, right_in_pocket

    # ----------------- Arms Crossed -----------------
    def detect_arms_crossed(self, landmarks):
        LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER = 15, 16, 11, 12
        lw, rw, ls, rs = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST], landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        chest_center_x, chest_center_y = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
        lw_near_chest = abs(lw.x - chest_center_x) < 0.27 and abs(lw.y - chest_center_y) < 0.27
        rw_near_chest = abs(rw.x - chest_center_x) < 0.27 and abs(rw.y - chest_center_y) < 0.27
        return lw_near_chest and rw_near_chest

    # ----------------- YOLO + Pose + Hand proximity Mobile Detection -----------------
    def detect_mobile_use_with_yolo(self, frame, landmarks):
        # Step 1: Check if both hands appear "closed" or close together
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb_frame)

        hands_close = False
        fingers_closed = False

        if hand_result.multi_hand_landmarks and len(hand_result.multi_hand_landmarks) >= 1:
            all_fingers_tips = []
            for hand_landmarks in hand_result.multi_hand_landmarks:
                tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # thumb & fingertips
                palm = hand_landmarks.landmark[0]
                avg_dist = np.mean([np.linalg.norm([t.x - palm.x, t.y - palm.y]) for t in tips])
                if avg_dist < 0.17:  
                    fingers_closed = True
                all_fingers_tips.extend(tips)

            if len(all_fingers_tips) >= 10:
                left_tips = np.array([[t.x, t.y] for t in all_fingers_tips[:5]])
                right_tips = np.array([[t.x, t.y] for t in all_fingers_tips[5:]])
                hand_dist = np.mean(np.linalg.norm(left_tips - right_tips, axis=1))
                if hand_dist < 0.17:
                    hands_close = True

        if hands_close and fingers_closed:
            return "Holding Mobile (Hands Together)"

        results = self.mobile_model(frame, verbose=False)
        has_mobile = False
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self.mobile_model.names[cls] == "cell phone":
                    has_mobile = True
                    break

        if not has_mobile:
            return "No Mobile Detected"

        LEFT_WRIST, RIGHT_WRIST, NOSE = 15, 16, 0
        lw, rw, nose = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST], landmarks[NOSE]

        def dist(a, b):
            return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

        hand_dist_pose = dist(lw, rw)
        lw_to_nose, rw_to_nose = dist(lw, nose), dist(rw, nose)

        if lw.y < 0.6 or rw.y < 0.6:
            return "Using Mobile (One Hand)"
        elif lw_to_nose < 0.1 or rw_to_nose < 0.1:
            return "On Call"
        else:
            return "Holding Mobile"

    def analyze_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb)
        face_result = self.face.process(rgb)
        self.total_frames += 1
        eye_contact_detected = False

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            left_in_pocket, right_in_pocket = self.detect_hand_in_pocket(landmarks)
            in_pocket = left_in_pocket or right_in_pocket
            crossed = self.detect_arms_crossed(landmarks)
            movement_value = np.std([[lm.x, lm.y] for lm in landmarks])
            movement = "Moving" if movement_value > 0.35 else "Still"
            mobile_state = self.detect_mobile_use_with_yolo(frame, landmarks)

            if ("Mobile" in mobile_state or "Call" in mobile_state) and "No" not in mobile_state:
                self.mobile_count += 1
            if in_pocket: 
                self.pocket_count += 1
                self.consecutive_pocket += 1
                self.max_consecutive_pocket = max(self.max_consecutive_pocket, self.consecutive_pocket)
            else:
                self.consecutive_pocket = 0
            if crossed: 
                self.crossed_count += 1
                self.consecutive_crossed += 1
                self.max_consecutive_crossed = max(self.max_consecutive_crossed, self.consecutive_crossed)
            else:
                self.consecutive_crossed = 0
            if movement == "Moving":
                self.moving_count += 1
        else:
            in_pocket = crossed = False
            movement = "Pose not detected"
            mobile_state = "No Mobile Detected"

        if face_result.multi_face_landmarks:
            eye_contact_detected = self.get_eye_contact(face_result.multi_face_landmarks[0])
            if eye_contact_detected:
                self.eye_contact_count += 1
                self.consecutive_no_eye_contact = 0
            else:
                self.consecutive_no_eye_contact += 1
                self.max_consecutive_no_eye_contact = max(self.max_consecutive_no_eye_contact, self.consecutive_no_eye_contact)
            eye_contact = "Eye Contact: Yes" if eye_contact_detected else "Eye Contact: No"
        else:
            eye_contact = "Eye Contact: Not Visible"

        pocket_state = "Hands in Pocket" if in_pocket else "Hands Free"
        arms_state = "Arms Crossed" if crossed else "Arms Open"
        analysis = f"{eye_contact}\n{pocket_state}\n{arms_state}\nMovement: {movement}\n{mobile_state}"
        self.analysis_queue.append(analysis)
        return analysis

    def generate_summary(self):
        if self.total_frames == 0:
            return "No frames analyzed."

        def percent(count): return round((count / self.total_frames) * 100, 1)
        fps_estimate = 30
        eye_contact_pct = percent(self.eye_contact_count)
        crossed_pct = percent(self.crossed_count)
        pocket_pct = percent(self.pocket_count)
        mobile_pct = percent(self.mobile_count)
        movement_pct = percent(self.moving_count)
        total_seconds = self.total_frames / fps_estimate

        return (
            f"=== BODY LANGUAGE SUMMARY ===\n"
            f"Video Duration: ~{total_seconds:.1f}s\n\n"
            f"Eye Contact: {eye_contact_pct}%\n"
            f"Arms Crossed: {crossed_pct}%\n"
            f"Hands in Pocket: {pocket_pct}%\n"
            f"Mobile Detected: {mobile_pct}%\n"
            f"Movement Detected: {movement_pct}%\n"
        )



def draw_vertical_text(frame, text, x, y_start, color=(0, 0, 0)):
    for i, line in enumerate(text.split('\n')):
        y = y_start + i * 25
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def process_video(video_path, output_path, text_output_path="analysis_report.txt"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Loading YOLOv8 model for mobile detection...")
    mobile_model = YOLO('yolov8n.pt')
    analyzer = ContinuousBodyLanguageAnalyzer(mobile_model=mobile_model)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        analysis_text = analyzer.analyze_frame(frame)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1, y1 = 5, 5  


        frame = draw_vertical_text(frame, analysis_text, x=x1 + 15, y_start=y1 + 40, color=(0, 0, 0))



        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(analyzer.generate_summary())

    print(f"✅ Analysis complete! Video saved to: {output_path}")
    print(f"📄 Report saved to: {text_output_path}")
    print("\n" + analyzer.generate_summary())



