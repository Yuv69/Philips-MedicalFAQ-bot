MEDIBOT - AI-Powered Medical FAQ Chatbot
An intelligent medical FAQ chatbot designed to provide accessible healthcare information through natural language understanding and voice interaction capabilities.
Overview
MEDIBOT is an AI-powered chatbot that uses semantic search and sentence embeddings to understand and respond to medical queries. The system is designed with accessibility in mind, particularly targeting elderly and non-tech-savvy users through voice input/output capabilities and an intuitive interface.
Features
Core Functionality

Natural Language Understanding: Utilizes sentence embeddings and semantic search for accurate query comprehension
Real-time Answer Retrieval: Fast response system using FAISS (Facebook AI Similarity Search) for efficient similarity matching
Voice Integration: Complete voice interaction support with speech-to-text input and text-to-speech output
Comprehensive Medical Database: Curated medical datasets covering common health questions and concerns

Accessibility Features

Voice Input: Google Speech-to-Text API integration for hands-free queries
Voice Output: Text-to-speech functionality using pyttsx3 for audio responses
User-Friendly Interface: Clean, intuitive Streamlit frontend designed for ease of use
Multilingual Support: Planned feature for broader accessibility

Technical Features

Semantic Search: Advanced NLP techniques for understanding medical terminology and context
FastAPI Backend: High-performance API for handling requests and responses
Vector Database: FAISS implementation for efficient similarity search
Sentence Transformers: State-of-the-art embeddings for medical text understanding

Technology Stack
Backend

Python: Core programming language
FastAPI: Modern web framework for building APIs
FAISS: Vector database for similarity search
Sentence Transformers: For generating semantic embeddings
pyttsx3: Text-to-speech conversion

Frontend

Streamlit: Web application framework for the user interface

APIs & Services

Google Speech API: Speech-to-text functionality
NLP Libraries: Natural language processing capabilities




Using the Chatbot

Text Input: Type your medical question in the input field
Voice Input: Click the microphone button and speak your question
Get Response: The system will process your query and provide relevant medical information
Listen to Response: Use the speaker button to hear the response

Example Queries

"What are the symptoms of flu?"
"How to manage diabetes?"
"What is normal blood pressure?"
"When should I see a doctor for headaches?"


Future Enhancements
Planned Features

Multilingual Support: Expand to support multiple languages
Medical Image Analysis: Integration with medical imaging capabilities
Appointment Scheduling: Connect with healthcare provider systems
Personalized Recommendations: User profile-based suggestions
Mobile Application: Native mobile app development
Integration APIs: Healthcare provider system integration

Technical Improvements

Enhanced NLP models for better medical understanding
Expanded medical database with specialist knowledge
Real-time learning from user interactions
Advanced voice recognition for medical terminology
