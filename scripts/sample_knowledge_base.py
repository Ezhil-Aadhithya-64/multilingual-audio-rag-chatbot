"""
Sample Knowledge Base Documents
Run this script to populate the vector database with sample documents
"""

from services.orchestrator import AgenticOrchestrator

# Sample documents covering various topics
SAMPLE_DOCUMENTS = [
    # Product Documentation
    {
        "text": """
        VoiceAssist Pro - Getting Started Guide
        
        VoiceAssist Pro is an AI-powered assistant that helps you with information retrieval,
        document search, and intelligent conversations. The system uses advanced RAG 
        (Retrieval-Augmented Generation) technology to provide accurate, grounded responses.
        
        Key Features:
        - Multilingual support for 20+ languages
        - Voice input and output capabilities
        - Intelligent document search and retrieval
        - Context-aware conversations with memory
        - Real-time response generation
        
        To get started, simply type your question in the chat interface or use voice input
        by clicking the microphone button.
        """,
        "metadata": {
            "source": "getting_started_guide",
            "language": "en",
            "topic": "documentation",
            "type": "guide"
        }
    },
    
    # Technical Support
    {
        "text": """
        Password Reset Instructions
        
        If you've forgotten your password, follow these steps:
        
        1. Click on the "Forgot Password" link on the login page
        2. Enter your registered email address
        3. Check your email for a password reset link (it may take 2-5 minutes)
        4. Click the link in the email (valid for 24 hours)
        5. Create a new strong password with at least 8 characters
        6. Confirm your new password
        7. Log in with your new credentials
        
        If you don't receive the email:
        - Check your spam/junk folder
        - Verify you entered the correct email address
        - Contact support if the issue persists
        
        For security reasons, password reset links expire after 24 hours.
        """,
        "metadata": {
            "source": "support_docs",
            "language": "en",
            "topic": "account_management",
            "type": "tutorial"
        }
    },
    
    # Product Features
    {
        "text": """
        Voice Input Feature
        
        VoiceAssist Pro supports voice input in multiple languages. To use voice input:
        
        1. Click the microphone icon in the chat interface
        2. Allow microphone access when prompted by your browser
        3. Speak clearly into your microphone
        4. Click stop when you're done speaking
        5. The system will automatically transcribe and process your query
        
        Supported Languages:
        - English, Spanish, French, German, Italian
        - Portuguese, Dutch, Polish, Russian
        - Japanese, Korean, Chinese (Mandarin)
        - Arabic, Hindi, Turkish, Vietnamese
        - And many more!
        
        The system automatically detects the language you're speaking and provides
        responses in the same language.
        """,
        "metadata": {
            "source": "feature_docs",
            "language": "en",
            "topic": "voice_features",
            "type": "feature"
        }
    },
    
    # FAQ
    {
        "text": """
        Frequently Asked Questions
        
        Q: How accurate are the AI responses?
        A: VoiceAssist Pro uses advanced RAG technology to ensure responses are grounded
        in your knowledge base. The system includes confidence scoring and hallucination
        detection to maintain high accuracy.
        
        Q: Can I use this in multiple languages?
        A: Yes! The system supports 20+ languages for both input and output. It automatically
        detects the language you're using.
        
        Q: Is my data secure?
        A: All conversations are encrypted and stored securely. We follow industry best
        practices for data protection and privacy.
        
        Q: How do I add my own documents?
        A: You can add documents through the admin panel or API. Documents are automatically
        processed, chunked, and indexed for efficient retrieval.
        
        Q: What happens if the system doesn't know the answer?
        A: The system will honestly indicate when it doesn't have enough information to
        answer your question, rather than making up an answer.
        """,
        "metadata": {
            "source": "faq",
            "language": "en",
            "topic": "general",
            "type": "faq"
        }
    },
    
    # API Documentation
    {
        "text": """
        API Integration Guide
        
        VoiceAssist Pro provides a RESTful API for integration with your applications.
        
        Base URL: https://api.voiceassistpro.com/v1
        
        Authentication:
        Include your API key in the Authorization header:
        Authorization: Bearer YOUR_API_KEY
        
        Main Endpoints:
        
        POST /query
        - Submit a text query
        - Request body: {"query": "your question", "session_id": "optional"}
        - Returns: {"response": "answer", "confidence": 0.95, "sources": [...]}
        
        POST /audio-query
        - Submit an audio file for transcription and processing
        - Content-Type: multipart/form-data
        - Returns: transcription and response
        
        GET /sessions/{session_id}
        - Retrieve conversation history
        - Returns: list of messages with metadata
        
        POST /ingest
        - Add documents to the knowledge base
        - Request body: {"documents": [...], "metadata": [...]}
        
        Rate Limits:
        - Free tier: 100 requests/hour
        - Pro tier: 1000 requests/hour
        - Enterprise: Custom limits
        """,
        "metadata": {
            "source": "api_docs",
            "language": "en",
            "topic": "api",
            "type": "technical"
        }
    },
    
    # Troubleshooting
    {
        "text": """
        Common Issues and Solutions
        
        Issue: Microphone not working
        Solution: 
        - Check browser permissions for microphone access
        - Ensure your microphone is properly connected
        - Try refreshing the page and allowing permissions again
        - Check if other applications are using the microphone
        
        Issue: Slow response times
        Solution:
        - Check your internet connection
        - Clear browser cache and cookies
        - Try using a different browser
        - Contact support if the issue persists
        
        Issue: Incorrect language detection
        Solution:
        - Speak more clearly and at a moderate pace
        - Manually select your language in settings
        - Ensure there's minimal background noise
        
        Issue: Can't find relevant information
        Solution:
        - Try rephrasing your question
        - Be more specific in your query
        - Check if the information exists in the knowledge base
        - Contact support to add missing documentation
        
        For additional help, contact our support team at support@voiceassistpro.com
        """,
        "metadata": {
            "source": "troubleshooting",
            "language": "en",
            "topic": "support",
            "type": "troubleshooting"
        }
    },
    
    # Pricing Information
    {
        "text": """
        VoiceAssist Pro Pricing Plans
        
        Free Tier:
        - 100 queries per month
        - Basic voice support
        - Email support
        - 1 GB document storage
        Price: $0/month
        
        Pro Tier:
        - 10,000 queries per month
        - Advanced voice features
        - Priority email support
        - 50 GB document storage
        - Custom branding
        - API access
        Price: $49/month
        
        Enterprise Tier:
        - Unlimited queries
        - Dedicated support team
        - Unlimited document storage
        - Custom integrations
        - SLA guarantees
        - On-premise deployment option
        Price: Custom pricing
        
        All plans include:
        - Multilingual support
        - Conversation memory
        - Regular updates
        - Security features
        
        Annual subscriptions receive a 20% discount.
        """,
        "metadata": {
            "source": "pricing",
            "language": "en",
            "topic": "pricing",
            "type": "commercial"
        }
    },
    
    # Privacy Policy
    {
        "text": """
        Privacy and Data Protection
        
        At VoiceAssist Pro, we take your privacy seriously.
        
        Data Collection:
        - We collect only the information necessary to provide our services
        - Conversation data is used to improve response quality
        - Voice recordings are processed and then deleted
        - No personal data is sold to third parties
        
        Data Storage:
        - All data is encrypted at rest and in transit
        - Conversations are stored for 90 days by default
        - You can request data deletion at any time
        - Backups are maintained for disaster recovery
        
        Data Usage:
        - Your data is used only to provide and improve our services
        - Aggregated, anonymized data may be used for analytics
        - We comply with GDPR, CCPA, and other privacy regulations
        
        Your Rights:
        - Access your data at any time
        - Request data deletion
        - Export your conversation history
        - Opt out of data collection for improvements
        
        For privacy concerns, contact privacy@voiceassistpro.com
        """,
        "metadata": {
            "source": "privacy_policy",
            "language": "en",
            "topic": "legal",
            "type": "policy"
        }
    }
]


def ingest_sample_documents():
    """Ingest sample documents into the knowledge base."""
    print("=" * 60)
    print("INGESTING SAMPLE KNOWLEDGE BASE")
    print("=" * 60)
    
    # Initialize orchestrator
    print("\n1. Initializing orchestrator...")
    orchestrator = AgenticOrchestrator()
    
    # Prepare documents and metadata
    documents = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
    
    # Ingest documents
    print(f"\n2. Ingesting {len(documents)} documents...")
    result = orchestrator.ingest_documents(
        documents=documents,
        metadatas=metadatas,
        chunk=True
    )
    
    if result["success"]:
        print(f"\n✅ Successfully ingested {result['num_documents']} documents")
        print(f"✅ Created {result['num_chunks']} chunks")
        print("\n" + "=" * 60)
        print("SAMPLE QUERIES TO TRY:")
        print("=" * 60)
        print("1. How do I reset my password?")
        print("2. What languages does VoiceAssist Pro support?")
        print("3. How much does the Pro tier cost?")
        print("4. How do I use voice input?")
        print("5. What are the API endpoints?")
        print("6. My microphone is not working, what should I do?")
        print("7. How is my data protected?")
        print("8. What's included in the free tier?")
        print("=" * 60)
    else:
        print(f"\n❌ Ingestion failed: {result.get('error')}")
    
    return result


if __name__ == "__main__":
    ingest_sample_documents()
