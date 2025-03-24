import os
import sys
import logging
import threading
import queue
import time
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, socketio):
        # Use a robust pre-trained emotion detection pipeline
        self.emotion_pipeline = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.socketio = socketio
        
        # Threading control
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        
        # Emotion labels
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 
            'optimism', 'pride', 'realization', 'relief', 'sadness', 
            'surprise', 'neutral'
        ]

    def detect_emotions(self, text):
        """
        Detect emotions from text with comprehensive probability scoring
        """
        try:
            # Ensure text is not empty and has meaningful content
            if not text or len(text.strip()) < 2:
                return {}

            # Perform emotion classification
            results = self.emotion_pipeline(text)[0]
            
            # Create probability dictionary
            emotion_scores = {}
            for result in results:
                emotion_scores[result['label']] = result['score']
            
            # Sort emotions by probability
            sorted_emotions = dict(
                sorted(
                    emotion_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]  # Top 5 emotions
            )
            
            return sorted_emotions
        
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return {}

    def speech_recognition_thread(self):
        """
        Continuous speech recognition thread
        """
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Speech recognition thread started...")

            while not self.stop_event.is_set():
                try:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=None, 
                        phrase_time_limit=3
                    )
                    self.audio_queue.put(audio)
                except Exception as e:
                    logger.error(f"Speech listening error: {e}")
                    time.sleep(1)

    def audio_processing_thread(self):
        """
        Audio processing and emotion detection thread
        """
        while not self.stop_event.is_set():
            try:
                # Get audio from queue with timeout
                audio = self.audio_queue.get(timeout=1)
                
                try:
                    # Recognize speech
                    text = self.recognizer.recognize_google(
                        audio, 
                        language='en-US'
                    )
                    
                    # Detect emotions
                    if text and len(text) > 2:
                        emotions = self.detect_emotions(text)
                        
                        if emotions:
                            # Emit via WebSocket
                            self.socketio.emit('emotion_update', {
                                'text': text,
                                'emotions': emotions
                            })
                        
                        logger.info(f"Recognized: {text}")
                        logger.info(f"Emotions: {emotions}")
                
                except sr.UnknownValueError:
                    # No speech detected
                    pass
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
            
            except queue.Empty:
                # No audio in queue
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def start(self):
        """
        Start emotion detection threads
        """
        self.stop_event.clear()
        
        # Start speech recognition thread
        threading.Thread(
            target=self.speech_recognition_thread, 
            daemon=True
        ).start()
        
        # Start audio processing thread
        threading.Thread(
            target=self.audio_processing_thread, 
            daemon=True
        ).start()

    def stop(self):
        """
        Stop emotion detection
        """
        self.stop_event.set()

# Flask Application Setup
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global emotion detector
emotion_detector = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global emotion_detector
    if emotion_detector is None:
        emotion_detector = EmotionDetector(socketio)
    emotion_detector.start()
    emit('status', {'message': 'Connected and listening'})

@socketio.on('disconnect')
def handle_disconnect():
    global emotion_detector
    if emotion_detector:
        emotion_detector.stop()

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Starting Voice Emotion Detector...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)