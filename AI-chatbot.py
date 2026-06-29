import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialize FastAPI App
app = FastAPI(title="AI Customer Support Chatbot", description="FAQ matching and NLP fallback with SQLite logging")

DB_FILE = "support_logs.db"

# --- 1. FAQ KNOWLEDGE BASE ---
# This serves as your customer support data source.
FAQ_BANK = {
    "What is your return policy?": "You can return any product within 30 days of purchase for a full refund. Items must be in original condition.",
    "How long does shipping take?": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "How can I track my order?": "Once your order ships, we will email you a tracking number. You can use it on our website's tracking page.",
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay.",
    "How do I contact human support?": "You can reach our human support team at support@example.com or call us at 1-800-555-0199."
}

# --- 2. DATABASE SETUP ---
def init_db():
    """Initializes the SQLite database for interaction logging."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_query TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            matched_by TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_interaction(query: str, response: str, match_type: str):
    """Logs customer interaction into SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO support_logs (customer_query, bot_response, matched_by) VALUES (?, ?, ?)",
        (query, response, match_type)
    )
    conn.commit()
    conn.close()

# --- 3. NLP INTENT / SIMILARITY SETUP ---
# We use a Sentence Transformers/Feature Extraction pipeline to mathematically 
# compare the customer's question to our FAQ bank.
print("Initializing NLP Subsystem for FAQ Matching...")
try:
    # A lightweight, highly efficient model for matching sentence meanings
    similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    print("NLP Model loaded successfully!")
except Exception as e:
    print(f"Error loading NLP model: {e}")
    raise e

def compute_embedding(text: str):
    """Converts text into a mathematical vector to understand its meaning."""
    outputs = similarity_model(text)
    # Mean pooling over the sequence dimension
    return torch.tensor(outputs[0]).mean(dim=0)

# Pre-calculate meanings for all our standard FAQ questions
FAQ_QUESTIONS = list(FAQ_BANK.keys())
FAQ_EMBEDDINGS = [compute_embedding(q) for q in FAQ_QUESTIONS]

def find_best_faq(customer_query: str, threshold=0.6):
    """Compares customer query against FAQs using cosine similarity."""
    query_vector = compute_embedding(customer_query)
    best_match_idx = -1
    highest_score = -1.0
    
    # Calculate how close the customer's question is to each FAQ
    for i, faq_vector in enumerate(FAQ_EMBEDDINGS):
        # Cosine similarity formula
        score = torch.nn.functional.cosine_similarity(query_vector.unsqueeze(0), faq_vector.unsqueeze(0)).item()
        if score > highest_score:
            highest_score = score
            best_match_idx = i
            
    # If the match is strong enough, return the answer
    if highest_score >= threshold:
        matched_question = FAQ_QUESTIONS[best_match_idx]
        return FAQ_BANK[matched_question], "FAQ_Match"
    
    # Fallback response if the bot doesn't know the answer
    fallback_text = "I'm sorry, I couldn't find an exact match for your question in our FAQs. Would you like me to connect you with a human support agent at support@example.com?"
    return fallback_text, "Fallback_No_Match"

# --- 4. DATA MODELS ---
class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    matched_by: str

# --- 5. API ENDPOINTS ---

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(payload: QueryRequest):
    customer_query = payload.message.strip()
    
    if not customer_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Route query through NLP matching engine
        bot_response, match_type = find_best_faq(customer_query)
        
        # Log to SQLite
        log_interaction(customer_query, bot_response, match_type)
        
        return QueryResponse(response=bot_response, matched_by=match_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal NLP Error: {str(e)}")

@app.get("/logs")
def get_logs(limit: int = 50):
    """Retrieve user interaction logs for business analysis."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, customer_query, bot_response, matched_by, timestamp FROM support_logs ORDER BY id DESC LIMIT ?", 
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return {
        "total_logged": len(rows),
        "logs": [
            {"id": r[0], "customer_query": r[1], "bot_response": r[2], "matched_by": r[3], "timestamp": r[4]}
            for r in rows
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)