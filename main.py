import spacy
from fastapi import FastAPI, Request
from keybert import KeyBERT
from pydantic import BaseModel

app = FastAPI(title="Lightweight Topic & Question Generator")

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")  # small & fast


class TextPayload(BaseModel):
    text: str


@app.post("/get_topics")
async def get_topics(payload: TextPayload):
    text = payload.text
    if not text:
        return {"error": "Text is required"}

    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
    )
    topics = [kw[0] for kw in keywords]
    return {"topics": topics}


@app.post("/get_questions")
async def get_questions(payload: TextPayload):
    text = payload.text
    if not text:
        return {"error": "Text is required"}

    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "DATE"):
                q = f"What is {ent.text} related to?"
                questions.append(q)
    if not questions:
        questions = [
            "Can you explain this topic?",
            "What are the key points discussed?",
        ]
    return {"questions": questions}
