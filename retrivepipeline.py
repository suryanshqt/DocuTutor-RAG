import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from indexingpipe import get_active_collection, QDRANT_URL

load_dotenv()

# --- Global model clients (cheap to init, no network call at startup) ---
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
embedding_models = OpenAIEmbeddings(model="text-embedding-3-large")


def get_vector_db() -> QdrantVectorStore:
    """
    Lazily connects to whichever collection is currently active.
    Raises a clear error if no PDF has been indexed yet.
    """
    collection = get_active_collection()
    if not collection:
        raise RuntimeError(
            "No document indexed yet. "
            "Please upload a PDF first via the /upload-pdf endpoint."
        )
    try:
        return QdrantVectorStore.from_existing_collection(
            url=QDRANT_URL,
            collection_name=collection,
            embedding=embedding_models
        )
    except Exception:
        raise RuntimeError(
            f"Collection '{collection}' not found in Qdrant. "
            "It may have been deleted. Please re-upload your PDF."
        )


# --- Data Structures for Quiz Output ---

class MCQ(BaseModel):
    question: str = Field(description="The multiple choice question")
    options: List[str] = Field(description="List of 4 possible options")
    correct_answer: str = Field(description="The exact text of the correct option")

class QuizPaper(BaseModel):
    mcqs: List[MCQ] = Field(description="List of 3 multiple choice questions")
    short_answers: List[str] = Field(description="List of 2 short answer questions")


def ask_tutor(user_query: str) -> dict:
    """Retrieves context and answers queries. Returns answer + cited page numbers."""
    vector_db = get_vector_db()
    search_results = vector_db.similarity_search(query=user_query, k=4)

    # Collect unique page numbers from retrieved chunks
    pages = sorted(set(
        str(r.metadata.get("page_label") or r.metadata.get("page", ""))
        for r in search_results
        if r.metadata.get("page_label") or r.metadata.get("page") is not None
    ))

    context = "\n\n".join(
        [
            f"Page Content: {result.page_content}\n"
            f"Page Number: {result.metadata.get('page_label', 'Unknown')}\n"
            f"File Location: {result.metadata.get('source', 'Unknown')}"
            for result in search_results
        ]
    )

    SYSTEM_PROMPT = f"""
You are an expert technical tutor. Answer the query using ONLY the provided context.
Do NOT include page numbers inside your answer text — they will be shown separately.
If the answer cannot be determined from the context, state that the information
is unavailable in the current index.

Context:
{context}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ])

    return {"answer": response.content, "pages": pages}


def generate_quiz(topic: str) -> dict:
    """Generates a structured JSON assessment based on the requested topic."""
    vector_db = get_vector_db()
    search_results = vector_db.similarity_search(query=topic, k=5)
    study_material = "\n\n".join([res.page_content for res in search_results])

    parser = JsonOutputParser(pydantic_object=QuizPaper)

    prompt = PromptTemplate(
        template="""You are an expert technical instructor creating an assessment.
Read the following excerpts and generate a question paper based strictly on this content.

{format_instructions}

TEXTBOOK EXCERPTS:
{text}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    quiz_chain = prompt | llm | parser
    return quiz_chain.invoke({"text": study_material})