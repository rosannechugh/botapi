import cv2
import numpy as np
import os
import google.generativeai as genai
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Configure GenAI API Key
genai.configure(api_key="YOUR_GOOGLE_GENAI_API_KEY")

difficulty_level = 2  # Default

def analyze_emotion(frame):
    """Detects face and determines emotion-based difficulty level."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        mouth_region = roi_gray[int(0.7 * h):h, :]
        eye_region = roi_gray[:int(0.3 * h), :]

        mouth_intensity = np.mean(mouth_region)
        eye_intensity = np.mean(eye_region)

        if mouth_intensity > 130 and eye_intensity < 80:
            return 1  # Angry
        elif mouth_intensity > 120:
            return 3  # Happy
        elif mouth_intensity < 80 and eye_intensity > 100:
            return 1  # Sad
        elif 85 < mouth_intensity < 120 and 85 < eye_intensity < 120:
            return 1  # Confused
        else:
            return 2  # Neutral

def create_prompt(difficulty, qtype, topic):
    """Generates the AI question prompt based on difficulty, type, and topic."""
    prompt = "Give me a "
    if difficulty == 1:
        prompt += "easy "
    elif difficulty == 2:
        prompt += "medium difficulty "
    elif difficulty == 3:
        prompt += "hard "

    prompt += f"{qtype} question based on {topic}."
    return prompt

def generate_question(difficulty_level):
    """Fetches a generated question from Google AI."""
    model = genai.GenerativeModel()
    chat_session = model.start_chat(history=[])

    prompt = create_prompt(difficulty_level, "Theory", "Statistics")
    response = chat_session.send_message(prompt)
    model_response = response.text

    try:
        Q = model_response.split('?')[0].split(':')[1] + '?'
        optA = model_response.split('.')[0].split('?')[1]
        keys = ['a', 'b', 'c', 'd', "correct"]
        values = [optA]
        values.extend(model_response.split('.')[1:4])
        values.append(model_response.split(':')[2].split('.')[0])
        return dict(zip(keys, values))
    except:
        return {"error": "Failed to parse AI response."}

@app.route('/data', methods=['GET'])
def get_data():
    """API Route to get the question data in JSON format."""
    data = generate_question(difficulty_level)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
