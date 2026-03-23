import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from indexingpipe import index_document
from retrivepipeline import ask_tutor, generate_quiz

app = FastAPI(title="DocuTutor API", version="1.0.0")

# Configure CORS so the HTML frontend can reach this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf")
async def upload_and_index(file: UploadFile = File(...)):
    """
    Receives a PDF file, saves it temporarily, indexes it into the
    Qdrant vector database, then removes the temp file.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are accepted."}

    # FIX: use a UUID prefix so concurrent uploads never collide
    # and path-traversal characters in filenames are neutralised.
    safe_name = f"tmp_{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(os.getcwd(), safe_name)

    try:
        # 1. Save upload to a safe temp path
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run the indexing pipeline — returns the new collection name
        collection_name = index_document(file_name=file_path)

        return {
            "status": "success",
            "message": f"'{file.filename}' has been indexed successfully.",
            "collection": collection_name
        }

    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process document: {str(e)}"}
    finally:
        # 3. Always clean up, even on failure
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/chat")
async def chat_with_tutor(question: str = Form(...)):
    """
    Returns a context-aware, page-cited answer from the indexed documents.
    """
    if not question.strip():
        return {"status": "error", "answer": "Please provide a non-empty question."}
    try:
        result = ask_tutor(question)
        return {"status": "success", "answer": result["answer"], "pages": result["pages"]}
    except RuntimeError as e:
        return {"status": "error", "answer": str(e), "pages": []}
    except Exception as e:
        return {"status": "error", "answer": f"System error: {str(e)}", "pages": []}


@app.post("/generate-quiz")
async def create_quiz(topic: str = Form(...)):
    """
    Generates a structured MCQ + short-answer assessment from indexed content.
    """
    if not topic.strip():
        return {"status": "error", "message": "Please provide a non-empty topic."}
    try:
        quiz_data = generate_quiz(topic)
        return {"status": "success", "quiz": quiz_data}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to generate quiz: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)