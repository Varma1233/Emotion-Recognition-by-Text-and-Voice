import os
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import numpy as np

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class EmotionAwareGeminiChatbot:
    def __init__(self):
        # Configure Google Gemini API
        genai.configure(api_key="AIzaSyCwT8adNatXy_xYWdHgBrWtYyk252c4h94")  # Replace with your actual API key
        
        # Initialize Gemini model with better configuration for empathetic responses
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.8,  # Slightly increased for more creative responses
                "top_p": 0.92,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Initialize chat session with better initial prompt
        self.chat = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["I want you to be an emotion-aware assistant that responds with empathy and understanding."],
                },
                {
                    "role": "model",
                    "parts": [
                        "I'll be your emotion-aware assistant. I'll carefully analyze the emotional tone of your messages "
                        "and respond with empathy and understanding. I'll acknowledge your feelings and provide support "
                        "tailored to your emotional state. Feel free to share whatever is on your mind."
                    ],
                },
            ]
        )
        
        # Enhanced Emotion Analysis
        try:
            from transformers import pipeline
            self.emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
            self.transformer_available = True
        except (ImportError, Exception) as e:
            logging.warning(f"Could not load transformer model: {e}")
            self.transformer_available = False
            
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Adding a confidence threshold for emotion detection
        self.emotion_confidence_threshold = 0.4
        
        # Track conversation history for context
        self.conversation_history = []
        self.emotion_history = []
        
        # Logging Setup with file output
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"chatbot_{time.strftime('%Y%m%d_%H%M%S')}.log"))
        console_handler = logging.StreamHandler()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Welcome Message
        self.welcome_message = (
            "Hello! I'm your emotion-aware assistant. "
            "I can detect how you're feeling and respond accordingly. "
            "Feel free to type naturally, and I'll do my best to understand both your words and emotions."
        )
    
    def analyze_emotions(self, text):
        """Enhanced emotion analysis using multiple methods for better accuracy."""
        try:
            combined_emotions = {}
            
            # Use transformer-based model if available
            if self.transformer_available:
                results = self.emotion_pipeline(text)
                transformer_emotions = {result["label"]: result["score"] for result in results}
                combined_emotions = transformer_emotions.copy()
                self.logger.info(f"Transformer emotions: {transformer_emotions}")
            
            # Add VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            self.logger.info(f"VADER sentiment: {vader_scores}")
            
            # If transformer not available or as enhancement
            if not self.transformer_available or len(combined_emotions) == 0:
                # Create basic emotions from VADER
                if vader_scores['compound'] >= 0.5:
                    combined_emotions['joy'] = 0.7 * vader_scores['pos']
                elif vader_scores['compound'] <= -0.5:
                    combined_emotions['sadness'] = 0.7 * vader_scores['neg']
                    combined_emotions['anger'] = 0.3 * vader_scores['neg']
                else:
                    combined_emotions['neutral'] = 0.8
            else:
                # Enhance transformer emotions with VADER
                if vader_scores['compound'] >= 0.5:
                    combined_emotions['joy'] = max(combined_emotions.get('joy', 0), 0.7 * vader_scores['pos'])
                elif vader_scores['compound'] <= -0.5:
                    combined_emotions['sadness'] = max(combined_emotions.get('sadness', 0), 0.7 * vader_scores['neg'])
                    combined_emotions['anger'] = max(combined_emotions.get('anger', 0), 0.3 * vader_scores['neg'])
            
            # Determine dominant emotion with confidence check
            if combined_emotions:
                dominant_emotion = max(combined_emotions, key=combined_emotions.get)
                dominant_score = combined_emotions[dominant_emotion]
            else:
                dominant_emotion = "neutral"
                dominant_score = 0.8
            
            # Fall back to "neutral" if confidence is low
            if dominant_score < self.emotion_confidence_threshold:
                if vader_scores['compound'] > 0.2:
                    dominant_emotion = "joy"
                    dominant_score = 0.5
                elif vader_scores['compound'] < -0.2:
                    dominant_emotion = "sadness"
                    dominant_score = 0.5
                else:
                    dominant_emotion = "neutral"
                    dominant_score = 0.8
            
            # Context-aware emotion smoothing using history
            if self.emotion_history:
                # Reduce emotional "jumping" by considering recent emotions
                recent_emotions = self.emotion_history[-3:] if len(self.emotion_history) >= 3 else self.emotion_history
                if dominant_emotion not in recent_emotions and dominant_score < 0.7:
                    # If new emotion is significantly different but not very strong, smooth the transition
                    most_common_recent = max(set(recent_emotions), key=recent_emotions.count)
                    if most_common_recent in combined_emotions:
                        # Blend with recent emotion if it's a drastic change
                        if combined_emotions[most_common_recent] > dominant_score * 0.7:
                            dominant_emotion = most_common_recent
                            dominant_score = combined_emotions[most_common_recent]
            
            # Update emotion history
            self.emotion_history.append(dominant_emotion)
            
            # Log detailed emotion analysis
            self.logger.info(f"Combined analysis: {combined_emotions}")
            self.logger.info(f"Detected emotion: {dominant_emotion} ({dominant_score:.4f})")
            
            return combined_emotions, dominant_emotion, dominant_score
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return {}, "neutral", 0.5
    
    def generate_gemini_response(self, user_input, emotions, dominant_emotion, confidence_score):
        """Generate contextual response using Gemini with enhanced emotion awareness."""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "text": user_input, "emotion": dominant_emotion})
            
            # Create a more detailed prompt for better emotional context
            emotion_prompt = (
                f"The user's message appears to express primarily {dominant_emotion.upper()} "
                f"(confidence: {confidence_score:.2f}). Their emotional profile shows:"
            )
            
            # Add detailed emotion breakdown
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:  # Only include significant emotions
                    emotion_prompt += f"\n- {emotion}: {score:.2f}"
            
            # Add conversation context
            if len(self.conversation_history) > 1:
                emotion_prompt += "\n\nRecent conversation context:"
                context_window = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
                for i, entry in enumerate(context_window):
                    if i < len(context_window) - 1:  # Skip the current message
                        emotion_prompt += f"\n- Previous {entry['role']}: {entry['text']} (emotion: {entry['emotion']})"
            
            # More specific guidance for the model
            emotion_prompt += (
                f"\n\nRespond with empathy to the user's message. Address their emotional state appropriately for "
                f"someone feeling {dominant_emotion}. Provide supportive and helpful responses that acknowledge "
                f"their feelings. Keep your response concise and natural."
                f"\n\nUser's message: {user_input}"
            )
            
            # Send to Gemini with emotional context
            response = self.chat.send_message(emotion_prompt)
            
            # Clean up the response if needed
            response_text = response.text.strip()
            
            # Store in conversation history
            self.conversation_history.append({"role": "assistant", "text": response_text, "emotion": "supportive"})
            
            return response_text
        except Exception as e:
            self.logger.error(f"Gemini response generation error: {e}")
            
            # Enhanced fallback responses
            fallback_responses = {
                "joy": [
                    "That sounds wonderful! I'm glad you're feeling positive. Would you like to share more about what's bringing you joy?",
                    "It's great to hear you're in good spirits! Those positive feelings are valuable - what's contributing to them?"
                ],
                "sadness": [
                    "I hear that you're feeling down. It's okay to feel this way, and I'm here to listen if you want to talk more.",
                    "I'm sorry you're feeling sad. Sometimes expressing these feelings can help. Would you like to share what's on your mind?"
                ],
                "anger": [
                    "I can sense your frustration. Taking a deep breath might help in the moment. Would you like to talk about what's bothering you?",
                    "I understand you're feeling upset. Your feelings are valid, and I'm here to listen if you want to discuss what's happening."
                ],
                "fear": [
                    "It sounds like you might be worried or anxious. Remember that acknowledging these feelings is an important first step.",
                    "I hear that you're feeling uncertain or afraid. Would talking through what's concerning you help?"
                ],
                "love": [
                    "Those warm feelings sound wonderful to experience. Would you like to share more about this connection?",
                    "It's beautiful to hear about these positive feelings. Would you like to tell me more about what's inspiring them?"
                ],
                "surprise": [
                    "That does sound unexpected! How do you feel about this surprise development?",
                    "Unexpected things can certainly catch us off guard. How are you processing this surprise?"
                ],
                "neutral": [
                    "Thank you for sharing that with me. Would you like to explore this topic further?",
                    "I appreciate you telling me about this. Is there anything specific you'd like to discuss?"
                ]
            }
            
            # Choose a random response from the appropriate category for variety
            responses = fallback_responses.get(dominant_emotion, fallback_responses["neutral"])
            return np.random.choice(responses)
    
    def process_message(self, user_input):
        """Process a user message and return an appropriate response."""
        try:
            # Check for empty input
            if not user_input or user_input.strip() == "":
                return "I didn't catch that. Could you please share your thoughts?"
            
            # Log the incoming message
            self.logger.info(f"Received message: {user_input}")
            
            # Analyze emotions in the message
            emotions, dominant_emotion, confidence_score = self.analyze_emotions(user_input)
            
            # Get appropriate response
            response = self.generate_gemini_response(user_input, emotions, dominant_emotion, confidence_score)
            
            # Log the response
            self.logger.info(f"Generated response: {response}")
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I'm having trouble processing your message right now. Could you try again?"
    
    def run_console(self):
        """Run the chatbot in console mode for testing."""
        print(self.welcome_message)
        
        try:
            while True:
                user_input = input("\nYou: ")
                
                # Exit command
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Thank you for chatting! Goodbye.")
                    break
                
                # Process the message
                response = self.process_message(user_input)
                print(f"\nChatbot: {response}")
        except KeyboardInterrupt:
            print("\nChat session ended.")
    
    def reset_conversation(self):
        """Reset the conversation history and emotion tracking."""
        self.conversation_history = []
        self.emotion_history = []
        self.chat = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["I want you to be an emotion-aware assistant that responds with empathy and understanding."],
                },
                {
                    "role": "model",
                    "parts": [
                        "I'll be your emotion-aware assistant. I'll carefully analyze the emotional tone of your messages "
                        "and respond with empathy and understanding. I'll acknowledge your feelings and provide support "
                        "tailored to your emotional state. Feel free to share whatever is on your mind."
                    ],
                },
            ]
        )
        self.logger.info("Conversation reset")
        return self.welcome_message
    
    def get_conversation_summary(self):
        """Get a summary of the conversation and emotion history."""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = "Conversation Summary:\n"
        summary += f"- Messages exchanged: {len(self.conversation_history)}\n"
        
        if self.emotion_history:
            # Calculate most common emotion
            emotions_count = {}
            for emotion in self.emotion_history:
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
            
            most_common = max(emotions_count.items(), key=lambda x: x[1])
            summary += f"- Most common emotion: {most_common[0]} (detected {most_common[1]} times)\n"
            
            # Emotion transitions
            summary += "- Emotion flow: " + " → ".join(self.emotion_history[-5:]) + "\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    chatbot = EmotionAwareGeminiChatbot()
    chatbot.run_console()