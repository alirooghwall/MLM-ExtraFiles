# backend/app.py
"""
Production-ready FastAPI backend for MLM + Study Coach AI (Dari).
Place this file at backend/app.py and deploy to Render (or run locally).

Requirements (put in backend/requirements.txt):
fastapi
uvicorn[standard]
httpx
pydantic
pymongo
python-multipart
python-dotenv
pytesseract        # optional, only if you want OCR on server
Pillow             # required by pytesseract
# If you plan to load models locally (NOT recommended on Render free): transformers accelerate torch
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime

# Optional OCR (install pytesseract + pillow if you want upload OCR)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ----------------------------
# Configuration (via ENV VARS)
# ----------------------------
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")               # Hugging Face token (if needed)
HF_SPACE_URL = os.getenv("HF_SPACE_URL")       # e.g. https://username-space.hf.space/run/predict
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "0") == "1"  # "1" to load local transformers model (only if you installed libs and have resources)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "")  # e.g. "tiiuae/falcon-7b-instruct"

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable must be set.")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlm-coach-backend")

# ----------------------------
# MongoDB Setup
# ----------------------------
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.get_database("mlm_ai_db")
users_col = db.get_collection("users")

# Ensure indexes for queries (user_id)
users_col.create_index("user_id", unique=True)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="MLM Study Coach API", version="1.0")

# ----------------------------
# Models
# ----------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    language: Optional[str] = "dari"  # keep for future multilingual support

class ChatResponse(BaseModel):
    response: str
    actions: Optional[List[Dict[str, Any]]] = None   # optional structured actions parsed from AI reply

class QuizStartRequest(BaseModel):
    user_id: str

class QuizAnswerRequest(BaseModel):
    user_id: str
    lesson_id: str
    question_id: int
    answer: str

class TaskUpdateRequest(BaseModel):
    user_id: str
    task_id: str
    status: str
    note: Optional[str] = None

# ----------------------------
# AI Integration Helpers
# ----------------------------
# If LOCAL_MODEL=True, you can load a transformers pipeline here (not provided by default to avoid heavy deps on Render)
local_pipeline = None
if LOCAL_MODEL:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        logger.info("Loading local model... this may take long and require a lot of memory.")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, device_map="auto")
        local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.exception("Failed to load local model: %s", e)
        local_pipeline = None
        LOCAL_MODEL = False

async def call_remote_hf_space(prompt: str, max_tokens: int = 400) -> str:
    """
    Call a Hugging Face Space that uses Gradio's /run/predict endpoint.
    The Space should accept a single text input and return text in data[0].
    """
    if not HF_SPACE_URL or not HF_TOKEN:
        raise RuntimeError("HF_SPACE_URL and HF_TOKEN must be set for remote AI calls.")

    payload = {"data": [prompt]}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HF_SPACE_URL, json=payload, headers=headers)
        r.raise_for_status()
        resp_json = r.json()
        # Gradio run/predict returns {"data": [...], "duration": ...}
        if isinstance(resp_json.get("data"), list) and len(resp_json["data"]) > 0:
            result = resp_json["data"][0]
            if isinstance(result, str):
                return result
            # Sometimes result may be dict/other; convert to string
            return json.dumps(result)
        raise RuntimeError("Unexpected HF Space response format: %s" % resp_json)

async def call_ai(prompt: str) -> str:
    """
    Central AI call function: chooses local pipeline vs remote HF Space.
    Returns generated text.
    """
    logger.debug("AI prompt (truncated): %s", (prompt[:1000] + "...") if len(prompt) > 1000 else prompt)
    if LOCAL_MODEL and local_pipeline:
        # local pipeline is synchronous; wrap in thread if necessary
        return local_pipeline(prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
    else:
        return await call_remote_hf_space(prompt)

# ----------------------------
# Prompt Builders
# ----------------------------
def build_system_prompt_dari(user_doc: dict, extra: Optional[str] = "") -> str:
    """
    Builds a system prompt in Dari that includes user progress and asks AI to respond step-by-step.
    Modify the template as you like to shape the persona.
    """
    progress = user_doc.get("progress", {})
    checklist = user_doc.get("checklist", [])
    lessons = user_doc.get("lessons", [])

    prompt = (
        "شما یک مربی و مشاور حرفه‌ای MLM به زبان دری هستید. "
        "همیشه به صورت مرحله به مرحله و با لحنی انگیزشی پاسخ دهید. "
        "اگر نیاز به آزمون دارید، سوالات را مرحله به مرحله ارائه کنید و پاسخ را ارزیابی کنید. "
        "اگر جواب‌ها نیاز به بروزرسانی چک‌لیست دارند، خروجی را به صورت JSON در پایان پاسخ بدهید "
        "(مثال: {\"actions\": [{\"action\":\"update_task\",\"task_id\":\"followup\",\"status\":\"completed\"}]})\n\n"
    )
    prompt += f"اطلاعات کاربر: progress={json.dumps(progress)}, checklist={json.dumps(checklist)}, lessons={json.dumps(lessons)}\n\n"
    if extra:
        prompt += extra + "\n\n"
    prompt += "وقتی کاربر پیام می‌دهد، اول پاسخ گام به گام بده، سپس اگر کاری لازم بود، یک شی JSON با کلید 'actions' برگردان."
    return prompt

# ----------------------------
# Utility: parse possible JSON actions from model output
# ----------------------------
JSON_ACTION_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)  # naive balanced braces regex (may be heavy)

def extract_json_actions(text: str) -> List[Dict[str, Any]]:
    """
    Try to find a JSON object in the model output and return actions list inside it.
    The model is instructed to append a JSON object like {"actions":[...]} at the end of its reply.
    """
    actions = []
    # simple approach: find last {...} block and try to parse
    matches = list(JSON_ACTION_RE.finditer(text))
    if not matches:
        return []
    for m in reversed(matches):
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "actions" in obj and isinstance(obj["actions"], list):
                return obj["actions"]
        except Exception:
            continue
    return []

# ----------------------------
# OCR Helper (optional)
# ----------------------------
async def ocr_file_to_text(file: UploadFile) -> str:
    if not OCR_AVAILABLE:
        raise HTTPException(status_code=400, detail="OCR not available on this server. Install pytesseract and pillow.")
    # Save to temp and run pytesseract
    contents = await file.read()
    from io import BytesIO
    img = Image.open(BytesIO(contents)).convert("RGB")
    text = pytesseract.image_to_string(img, lang='fas+eng')  # 'fas' might not be perfect for Dari; adjust as needed
    return text

# ----------------------------
# API Endpoints
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint.
    - Retrieves user memory
    - Builds prompt in Dari
    - Calls AI
    - Parses any structured 'actions' the model returns
    - Updates DB based on safe actions
    """
    user = users_col.find_one({"user_id": req.user_id})
    if not user:
        # initialize default user doc
        user = {
            "user_id": req.user_id,
            "created_at": datetime.utcnow(),
            "lessons": [],
            "checklist": [],
            "progress": {}
        }
        users_col.insert_one(user)

    system_prompt = build_system_prompt_dari(user)
    full_prompt = f"{system_prompt}\nUser: {req.message}\nAI:"

    try:
        ai_text = await call_ai(full_prompt)
    except httpx.HTTPStatusError as e:
        logger.exception("HF Space call failed: %s", e)
        raise HTTPException(status_code=502, detail="AI service error.")
    except Exception as e:
        logger.exception("AI call error: %s", e)
        raise HTTPException(status_code=500, detail="AI internal error.")

    # Extract actions if provided by model
    actions = extract_json_actions(ai_text)

    # Execute safe backend actions (we only allow limited action types to avoid security risks)
    executed_actions = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("action") == "update_task":
            task_id = action.get("task_id")
            status = action.get("status")
            note = action.get("note", "")
            # apply update safely: find or create task in checklist
            checklist = user.get("checklist", [])
            found = False
            for t in checklist:
                if t.get("task_id") == task_id:
                    t["status"] = status
                    t["note"] = note
                    found = True
                    break
            if not found:
                checklist.append({"task_id": task_id, "status": status, "note": note})
            user["checklist"] = checklist
            users_col.update_one({"user_id": req.user_id}, {"$set": {"checklist": checklist}})
            executed_actions.append({"action": "update_task", "task_id": task_id, "status": status})
        elif action.get("action") == "update_progress":
            key = action.get("key")
            value = action.get("value")
            user_progress = user.get("progress", {})
            user_progress[key] = value
            user["progress"] = user_progress
            users_col.update_one({"user_id": req.user_id}, {"$set": {"progress": user_progress}})
            executed_actions.append({"action": "update_progress", "key": key, "value": value})
        else:
            # unknown action: ignore or log
            logger.info("Unknown action from AI (ignored): %s", action)

    # Save "last_interaction" log for analytics
    users_col.update_one({"user_id": req.user_id}, {"$set": {"last_message": req.message, "last_ai_reply": ai_text, "updated_at": datetime.utcnow()}})

    # For response, strip any trailing JSON from ai_text so user sees only natural text
    display_text = ai_text
    if actions:
        # attempt to remove last JSON block from display text
        last_json = json.dumps({"actions": actions}, ensure_ascii=False)
        if display_text.strip().endswith(last_json):
            display_text = display_text[:display_text.rfind(last_json)].strip()

    return ChatResponse(response=display_text, actions=executed_actions or None)

# ----------------------------
# Quiz endpoints (simple implementation)
# ----------------------------
@app.post("/quiz/start")
async def quiz_start(req: QuizStartRequest):
    """
    Generate a short quiz about the user's last lesson.
    """
    user = users_col.find_one({"user_id": req.user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    last_lesson = user.get("lessons", [])[-1] if user.get("lessons") else "هیچ درسی ذخیره نشده است."
    prompt = (
        f"شما یک مربی درس هستید. لطفاً 3 سوال کوتاه چندگزینه‌ای در مورد درس زیر بنویسید "
        f"برای سنجش یادگیری کاربر. درس: {last_lesson}\n\n"
        "هر سوال را با برچسب Q<number> و گزینه‌ها را با A/B/C/D نشان بده و شماره سوال را هم قرار بده."
    )
    try:
        ai_text = await call_ai(prompt)
    except Exception as e:
        logger.exception("Quiz AI error: %s", e)
        raise HTTPException(status_code=502, detail="AI service error.")

    # Store quiz temporarily in user doc with generated timestamp
    quiz_doc = {"quiz_id": f"quiz_{datetime.utcnow().timestamp()}", "questions_text": ai_text, "created_at": datetime.utcnow(), "answers": {}}
    users_col.update_one({"user_id": req.user_id}, {"$set": {"current_quiz": quiz_doc}})

    return {"quiz": ai_text, "quiz_id": quiz_doc["quiz_id"]}

@app.post("/quiz/answer")
async def quiz_answer(req: QuizAnswerRequest):
    """
    User answers a quiz question - ask AI to grade and provide feedback.
    """
    user = users_col.find_one({"user_id": req.user_id})
    if not user or not user.get("current_quiz"):
        raise HTTPException(status_code=404, detail="No active quiz found for this user")

    # get quiz text to provide context
    quiz = user["current_quiz"]
    # Build prompt to grade the specific answer
    prompt = (
        f"این متن سوالات و گزینه‌ها است:\n{quiz['questions_text']}\n\n"
        f"سوال شماره {req.question_id}، پاسخ کاربر: {req.answer}\n"
        "لطفاً پاسخ را با استدلال کوتاه تصحیح کن و امتیاز (0 یا 1) بده. "
        "پیشنهاد کن اگر اشتباه است چه کاری باید تمرین کند."
    )
    try:
        ai_text = await call_ai(prompt)
    except Exception as e:
        logger.exception("Quiz grade AI error: %s", e)
        raise HTTPException(status_code=502, detail="AI service error.")

    # update stored quiz answers
    users_col.update_one({"user_id": req.user_id}, {"$set": {f"current_quiz.answers.{req.question_id}": {"answer": req.answer, "feedback": ai_text}}})

    return {"feedback": ai_text}

# ----------------------------
# Task update endpoint (frontend calls this to update checklist)
# ----------------------------
@app.post("/task/update")
async def task_update(req: TaskUpdateRequest):
    user = users_col.find_one({"user_id": req.user_id})
    if not user:
        # init user
        user = {"user_id": req.user_id, "lessons": [], "checklist": [], "progress": {}}
        users_col.insert_one(user)

    checklist = user.get("checklist", [])
    updated = False
    for t in checklist:
        if t.get("task_id") == req.task_id:
            t["status"] = req.status
            if req.note:
                t["note"] = req.note
            updated = True
            break
    if not updated:
        checklist.append({"task_id": req.task_id, "status": req.status, "note": req.note or ""})

    users_col.update_one({"user_id": req.user_id}, {"$set": {"checklist": checklist, "updated_at": datetime.utcnow()}})
    return {"ok": True, "checklist": checklist}

# ----------------------------
# File upload (image/pdf) endpoint
# ----------------------------
@app.post("/upload")
async def upload_file(user_id: str, background: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept an image or PDF, run OCR (if available), save summary to user's notes,
    and optionally trigger AI analysis in background.
    """
    user = users_col.find_one({"user_id": user_id})
    if not user:
        user = {"user_id": user_id, "lessons": [], "checklist": [], "progress": {}}
        users_col.insert_one(user)

    # read content
    content = await file.read()
    # save file to a storage location if needed (omitted here for simplicity), e.g., AWS S3 or local disk.

    # If OCR available and file is image:
    extracted_text = ""
    if OCR_AVAILABLE and file.content_type.startswith("image/"):
        try:
            # run OCR synchronously (could be heavy), but we wrap in a background task if needed
            from io import BytesIO
            img = Image.open(BytesIO(content)).convert("RGB")
            extracted_text = pytesseract.image_to_string(img, lang='fas+eng')
        except Exception as e:
            logger.exception("OCR failed: %s", e)
            extracted_text = ""

    # Save short summary into user notes
    notes = user.get("notes", [])
    notes.append({"filename": file.filename, "summary": extracted_text[:2000], "uploaded_at": datetime.utcnow()})
    users_col.update_one({"user_id": user_id}, {"$set": {"notes": notes}})

    # Optionally analyze with AI in background (e.g., produce suggested reply)
    async def analyze_and_store(user_id_local, text):
        if not text:
            return
        prompt = build_system_prompt_dari(user) + f"\nUser uploaded a file with text:\n{text}\nPlease summarize and suggest next steps."
        try:
            ai_text = await call_ai(prompt)
            users_col.update_one({"user_id": user_id_local}, {"$push": {"ai_file_summaries": {"text": ai_text, "at": datetime.utcnow()}}})
        except Exception:
            logger.exception("Background AI analysis failed.")

    background.add_task(analyze_and_store, user_id, extracted_text)

    return {"ok": True, "summary": extracted_text[:1000]}

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
async def health():
    # quick DB ping
    try:
        client.admin.command("ping")
    except Exception as e:
        return {"ok": False, "db": False, "error": str(e)}
    return {"ok": True}
