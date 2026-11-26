
import os
import json
import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")

    text = ""

    for page in pdf_document:
        text += page.get_text()

    pdf_document.close()

    return text


@app.post("/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_data = await resume.read()

    try:
        resume_text = extract_text_from_pdf(pdf_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="No text found in PDF")

    prompt = f"""
    You are an expert recruitment and resume analysis system.

    Analyze the candidate's resume against the job description.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}

    Evaluate the candidate based on:
    - Technical skills
    - Relevant experience
    - Projects
    - Education
    - Job requirements

    Return your response in JSON format with the following structure:

    {{
        "match_percentage": 75,
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Docker", "AWS"],
        "strengths": ["Strong Python experience"],
        "improvements": ["Learn Docker"],
        "summary": "Short explanation of the match"
    }}

    match_percentage must be an integer between 0 and 100.
    Do not return anything outside the JSON.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        result = json.loads(response.text)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini analysis failed: {str(e)}"
        )
