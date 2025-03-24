import os
import logging
import traceback

# Import required libraries
try:
    import google.generativeai as genai
    from transformers import pipeline
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install google-generativeai transformers nltk flask flask-cors")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK resources: {e}")

class EmotionAwareGeminiChatbot:
    def __init__(self, api_key):
        try:
            # Configure Google Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize Gemini model
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.92,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Initialize chat session
            self.chat = self.model.start_chat(history=[])
            
            # Emotion Analysis Setup
            self.emotion_pipeline = pipeline("text-classification", 
                model="bhadresh-savani/bert-base-uncased-emotion")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.emotion_confidence_threshold = 0.4
            
            # Conversation History
            self.conversation_history = []
            self.emotion_history = []
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def analyze_emotions(self, text):
        """Enhanced emotion analysis with robust error handling."""
        try:
            # Transformer-based emotion model
            results = self.emotion_pipeline(text)
            transformer_emotions = {result["label"]: result["score"] for result in results}
            
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Combine both analysis methods
            combined_emotions = transformer_emotions.copy()
            
            # Map VADER sentiment to emotion categories
            if vader_scores['compound'] >= 0.5:
                combined_emotions['joy'] = max(combined_emotions.get('joy', 0), 0.7 * vader_scores['pos'])
            elif vader_scores['compound'] <= -0.5:
                combined_emotions['sadness'] = max(combined_emotions.get('sadness', 0), 0.7 * vader_scores['neg'])
                combined_emotions['anger'] = max(combined_emotions.get('anger', 0), 0.3 * vader_scores['neg'])
            
            # Determine dominant emotion
            dominant_emotion = max(combined_emotions, key=combined_emotions.get) if combined_emotions else "neutral"
            dominant_score = combined_emotions.get(dominant_emotion, 0.5)
            
            # Fall back to "neutral" if confidence is low
            if dominant_score < self.emotion_confidence_threshold:
                dominant_emotion = "neutral"
                dominant_score = 0.8
            
            # Update emotion history
            self.emotion_history.append(dominant_emotion)
            
            return combined_emotions, dominant_emotion, dominant_score
        
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            logger.error(traceback.format_exc())
            return {}, "neutral", 0.5
    
    def generate_response(self, user_input, emotions, dominant_emotion, confidence_score):
        """Generate contextual response using Gemini with emotion awareness."""
        try:
            # Create emotion-aware prompt
            emotion_prompt = (
                f"The user's message expresses {dominant_emotion.upper()} emotion "
                f"(confidence: {confidence_score:.2f}). Emotional profile:"
            )
            
            # Add emotion details
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:
                    emotion_prompt += f"\n- {emotion}: {score:.2f}"
            
            emotion_prompt += f"\n\nRespond empathetically to: {user_input}"
            
            # Send to Gemini
            response = self.chat.send_message(emotion_prompt)
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback responses
            fallback_responses = {
                "joy": "That's wonderful! I'm glad you're feeling positive.",
                "sadness": "I hear that you're feeling down. It's okay to feel this way.",
                "anger": "I understand you're frustrated. Let's talk about what's bothering you.",
                "neutral": "Thank you for sharing. What would you like to discuss?"
            }
            
            return fallback_responses.get(dominant_emotion, fallback_responses["neutral"])

# Flask App Setup
app = Flask(__name__)
CORS(app)

# API Key (REPLACE WITH YOUR ACTUAL API KEY)
GEMINI_API_KEY = 'AIzaSyCwT8adNatXy_xYWdHgBrWtYyk252c4h94'  # Replace with your actual key

# Initialize chatbot
try:
    chatbot = EmotionAwareGeminiChatbot(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Chatbot initialization failed: {e}")
    print("Failed to initialize chatbot. Check your API key.")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        
        # Analyze emotions
        emotions, dominant_emotion, confidence_score = chatbot.analyze_emotions(user_message)
        
        # Generate response
        bot_response = chatbot.generate_response(user_message, emotions, dominant_emotion, confidence_score)
        
        # Return response with emotion analysis
        return jsonify({
            'response': bot_response,
            'emotion': dominant_emotion,
            'emotion_details': {k: round(v, 2) for k, v in emotions.items() if v > 0.1}
        })
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'response': "I'm sorry, there was an error processing your message.",
            'emotion': 'error',
            'emotion_details': {}
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)