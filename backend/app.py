import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# --- MongoDB Setup ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client.mlm_ai_db
users_collection = db.users

# --- Hugging Face AI Setup ---
login(token=os.getenv("HF_TOKEN"))

model_name = "tiiuae/falcon-7b-instruct"  # smaller, fast for free tier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- FastAPI App ---
app = FastAPI()

# --- Request Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str

# --- Chat Endpoint ---
@app.post("/chat")
def chat(request: ChatRequest):
    user = users_collection.find_one({"user_id": request.user_id})
    
    # Initialize user if new
    if not user:
        user = {"user_id": request.user_id, "lessons": [], "checklist": [], "progress": {}}
        users_collection.insert_one(user)
    
    # Build AI prompt
    system_prompt = f"""
    You are a Dari-speaking MLM coach and study mentor.
    User progress: {user['progress']}
    Checklist: {user['checklist']}
    Last lessons: {user['lessons']}
    Respond step-by-step and motivate the user.
    User: {request.message}
    AI:
    """
    
    # Generate AI response
    response = generator(system_prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
    # Optionally trim prompt repetition
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()
    
    # Example: Update progress (MVP)
    user['progress']['last_message'] = request.message
    users_collection.update_one({"user_id": request.user_id}, {"$set": user})
    
    return {"response": response}
