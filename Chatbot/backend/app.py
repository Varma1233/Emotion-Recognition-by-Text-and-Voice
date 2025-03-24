from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
from chatbot import EmotionAwareGeminiChatbot

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])  # Allow requests from frontend server

# Initialize the chatbot
chatbot = EmotionAwareGeminiChatbot()

# Global variables for thread-safe chatbot interaction
chatbot_lock = threading.Lock()

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint to handle user input and return chatbot response."""
    try:
        data = request.json
        user_input = data.get("message")

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        # Use a lock to ensure thread-safe interaction with the chatbot
        with chatbot_lock:
            # Analyze emotions
            emotions, dominant_emotion, confidence_score = chatbot.analyze_emotions(user_input)

            # Generate response
            response = chatbot.generate_gemini_response(user_input, emotions, dominant_emotion, confidence_score)

            # Return the response
            return jsonify({
                "response": response,
                "emotion": dominant_emotion,
                "confidence": confidence_score,
                "emotion_breakdown": emotions
            })

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route("/start", methods=["GET"])
def start_chat():
    """Endpoint to start the chatbot session and get the welcome message."""
    try:
        with chatbot_lock:
            return jsonify({"response": chatbot.welcome_message})
    except Exception as e:
        logging.error(f"Error in /start endpoint: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)