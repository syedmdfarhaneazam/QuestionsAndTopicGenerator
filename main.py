from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI(title="Topic & Question Generator API")

question_generator = pipeline("text2text-generation", model="t5-base")
topic_generator = pipeline("text2text-generation", model="t5-base")


@app.post("/get_questions")
async def get_questions(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Text is required"}

    prompt = f"generate questions: {text}"
    result = question_generator(
        prompt, max_length=80, num_return_sequences=3, do_sample=True
    )
    questions = [r["generated_text"] for r in result]
    return {"questions": questions}


@app.post("/get_topics")
async def get_topics(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "Text is required"}

    prompt = f"extract important topics: {text}"
    result = topic_generator(
        prompt, max_length=50, num_return_sequences=3, do_sample=True
    )
    topics = [r["generated_text"] for r in result]
    return {"topics": topics}
