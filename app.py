from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from pymongo import MongoClient
import bcrypt
import cv2
import mediapipe as mp
import keyboard
import threading

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Establish MongoDB connection
client = MongoClient('mongodb+srv://Boomer:Boomer2004@cluster0.yp7ed.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['gesture_controlled_car_racing_game']
user_collection = db['user_profile']

# MediaPipe and Key Mapping Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Store key status
active_keys = {"right": None, "left": None}

# Threading to handle the prototype logic
process_thread = None
running = False


def load_gesture_mappings(user_email):
    """Load gesture-key mappings from MongoDB for the current user."""
    user_data = user_collection.find_one({"email": user_email})
    if user_data and "gesture_mappings" in user_data:
        return user_data["gesture_mappings"]
    else:
        print("No gesture mapping found for the user!")
        return {}


def recognize_gesture(hand_landmarks):
    """Recognizes a hand gesture based on finger positions."""
    landmarks = hand_landmarks.landmark

    fingers = []
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky fingertip indices
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:  # Fingertip above base
            fingers.append(1)
        else:
            fingers.append(0)

    # Hand tilt detection using wrist and finger base positions
    wrist_x = landmarks[0].x
    middle_base_x = landmarks[9].x
    tilt_threshold = 0.05

    # Gesture Conditions
    if fingers == [1, 1, 0, 0]:
        return "Victory"
    elif fingers == [1, 1, 1, 0]:
        return "Three Fingers Up"
    elif fingers == [1, 1, 1, 1]:
        if wrist_x < middle_base_x - tilt_threshold:
            return "Open Palm Tilted Left"
        elif wrist_x > middle_base_x + tilt_threshold:
            return "Open Palm Tilted Right"
        return "Open Palm"
    elif fingers == [0, 0, 0, 0]:
        return "Fist"

    return None


def run_gesture_detection(user_email):
    """Main gesture detection and key mapping logic."""
    global active_keys, running

    # Load user's gesture mappings
    GESTURE_KEY_MAPPING = load_gesture_mappings(user_email)

    # Start webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        detected_keys = {"right": None, "left": None}

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handedness = "left" if handedness_info.classification[0].label.lower() == "left" else "right"
                gesture = recognize_gesture(hand_landmarks)

                if gesture and gesture in GESTURE_KEY_MAPPING:
                    detected_keys[handedness] = GESTURE_KEY_MAPPING[gesture][handedness]

        # Handle key presses and releases
        for hand in ["right", "left"]:
            detected_key = detected_keys[hand]
            active_key = active_keys[hand]

            if detected_key and detected_key != active_key:
                if active_key:
                    keyboard.release(active_key)  # Release previous key
                keyboard.press(detected_key)  # Hold new key
                active_keys[hand] = detected_key

            elif not detected_key and active_key:
                keyboard.release(active_key)
                active_keys[hand] = None

        cv2.imshow("Hand Gesture Game Control", frame)

        if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to exit
            break

    # Release any active keys before exiting
    for key in active_keys.values():
        if key:
            keyboard.release(key)

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def landing_page():
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        user_data = user_collection.find_one({"email": session['user']})
        gesture_mappings = user_data.get("gesture_mappings", {})
        return render_template('index.html', gesture_mappings=gesture_mappings)
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = user_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user'] = email
            return redirect(url_for('dashboard'))
        return "Invalid credentials", 401
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        if user_collection.find_one({"email": email}):
            return "User already exists", 409
        user_collection.insert_one({"email": email, "password": hashed_password, "gesture_mappings": {}})
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/update_mappings', methods=['POST'])
def update_mappings():
    if 'user' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    data = request.json
    if data:
        user_collection.update_one(
            {"email": session['user']},
            {'$set': {"gesture_mappings": data}},
            upsert=True
        )
        return jsonify({'status': 'success'}), 200
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400


@app.route('/run_prototype', methods=['POST'])
def run_prototype():
    global process_thread, running
    if 'user' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    if running:
        return jsonify({'status': 'error', 'message': 'Prototype is already running'}), 400

    user_email = session['user']
    running = True
    process_thread = threading.Thread(target=run_gesture_detection, args=(user_email,))
    process_thread.start()
    return jsonify({'status': 'running'}), 200


@app.route('/stop_prototype', methods=['POST'])
def stop_prototype():
    global running
    if 'user' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    if running:
        running = False
        return jsonify({'status': 'stopped'}), 200

    return jsonify({'status': 'error', 'message': 'No process running'}), 400


if __name__ == '__main__':
    app.run(debug=True)
