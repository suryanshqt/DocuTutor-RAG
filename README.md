# DocuTutor Engine — RAG PDF Tutor (FastAPI + LangChain + Qdrant)

DocuTutor is a **Retrieval-Augmented Generation (RAG)** system that turns a PDF into a queryable knowledge base.
It supports:

- **Indexing**: PDF → pages → chunks → embeddings → **Qdrant** collection
- **Tutoring**: user question → similarity search → cited context → grounded answer
- **Assessment**: topic → retrieved excerpts → JSON-structured quiz generation

This README is written to be **interview-friendly**: it explains design decisions, data flow, and API contracts.

## Tech stack

- **FastAPI**: HTTP API (`/upload-pdf`, `/chat`, `/generate-quiz`)
- **LangChain**:
  - `PyPDFLoader` for page-level parsing
  - `RecursiveCharacterTextSplitter` for chunking
  - `QdrantVectorStore` for vector persistence + retrieval
  - `ChatOpenAI` for answer generation
- **Qdrant**: local vector DB via Docker Compose
- **OpenAI embeddings**: `text-embedding-3-large` (dimension inferred by LangChain)
- **LLM**: `gpt-4o-mini` (low-latency tutoring + quiz generation)

## Architecture (high-level)

**Frontend (static)**

- `index.html` + `style.css` + `script.js`
- Calls FastAPI directly over HTTP

**Backend (FastAPI)**

- `main.py`
  - uploads PDF
  - invokes the indexing pipeline
  - exposes chat + quiz endpoints

**Indexing pipeline**

- `indexingpipe.py`
  1. Load PDF pages
  2. Split into chunks
  3. Embed chunks
  4. Write vectors to Qdrant collection
  5. Persist “active collection” to `.active_collection`

**Retrieval + generation pipeline**

- `retrivepipeline.py`
  - lazy-connects to the active Qdrant collection
  - similarity search for context
  - LLM generation grounded in retrieved context
  - returns **answer + cited page numbers**

## Data flow (sequence)

### 1) Indexing flow

1. `POST /upload-pdf` (multipart PDF)
2. Server stores upload to a safe temp file (`tmp_<uuid>.pdf`)
3. `index_document(file_name=<tmp.pdf>)`:
   - deletes previous collection (single-active-document design)
   - loads pages via `PyPDFLoader`
   - chunking via `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)`
   - embeddings via `OpenAIEmbeddings(text-embedding-3-large)`
   - persists to Qdrant collection derived from filename
4. Active collection name persisted to `.active_collection`
5. Temp file removed

### 2) Chat flow

1. `POST /chat` with form field `question`
2. Backend runs `ask_tutor(question)`:
   - `similarity_search(k=4)` against active collection
   - builds a strict “answer using ONLY context” system prompt
   - calls the LLM
3. Returns:
   - `answer` (no page numbers embedded in text)
   - `pages` (deduped list, derived from chunk metadata)

### 3) Quiz flow

1. `POST /generate-quiz` with form field `topic`
2. Backend runs `generate_quiz(topic)`:
   - `similarity_search(k=5)`
   - uses `JsonOutputParser(pydantic_object=QuizPaper)` to enforce shape
3. Returns typed JSON quiz: MCQs + short answers

## Local setup

### 1) Start Qdrant

```bash
cd DocuTutor_Engine
docker compose up -d
```

Expected: Qdrant reachable at `http://localhost:6333`.

### 2) Environment variables

```bash
cd DocuTutor_Engine
cp .env.example .env
```

Set:

- `OPENAI_API_KEY=...`

### 3) Install dependencies

If you want a single environment for the whole repo:

```bash
pip install -r requirements.txt
```

Or install only DocuTutor deps:

```bash
pip install -r DocuTutor_Engine/requirements.txt
```

### 4) Run the API

```bash
cd DocuTutor_Engine
python main.py
```

Default: `http://127.0.0.1:8000`

### 5) Run the UI

Open `DocuTutor_Engine/index.html` in the browser.

## API contracts (technical)

### `POST /upload-pdf`

**Content-Type**: `multipart/form-data`

- field: `file` (PDF)

Response (success):

```json
{
  "status": "success",
  "message": "'MyBook.pdf' has been indexed successfully.",
  "collection": "mybook"
}
```

### `POST /chat`

**Content-Type**: `multipart/form-data`

- field: `question` (string)

Response (success):

```json
{
  "status": "success",
  "answer": "...",
  "pages": ["12", "13"]
}
```

### `POST /generate-quiz`

**Content-Type**: `multipart/form-data`

- field: `topic` (string)

Response (success):

```json
{
  "status": "success",
  "quiz": {
    "mcqs": [
      {
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_answer": "..."
      }
    ],
    "short_answers": ["...", "..."]
  }
}
```

## Key design decisions (what to highlight in interviews)

1. **Single-active-document collections**
   - The system keeps one “active” Qdrant collection tracked via `.active_collection`.
   - Pros: simple mental model, no multi-tenant routing, low ops overhead.
   - Tradeoff: multi-document support would require collection routing per user/session.

2. **Chunking strategy**
   - `chunk_size=1000`, `chunk_overlap=400` balances retrieval recall vs token budget.
   - Overlap reduces boundary information loss (definitions spanning pages/sections).

3. **Grounded answering**
   - System prompt explicitly restricts the model to provided context.
   - Page references are returned separately to avoid “hallucinated citations”.

4. **Structured output for quizzes**
   - Pydantic schema + JSON parser enforces response shape.
   - This is a production-friendly pattern for post-processing and UI rendering.

5. **Safe upload handling**
   - Uploaded PDFs are stored as `tmp_<uuid>.pdf` to avoid name collisions and path traversal.

## Troubleshooting

- **`No document indexed yet`**
  - Upload a PDF first (indexing must run before chat/quiz).

- **`Collection '<name>' not found in Qdrant`**
  - Qdrant may have been reset or the collection deleted. Re-upload the PDF.

- **Docker/Qdrant not running**
  - Ensure Docker Desktop is running.
  - Re-run `docker compose up -d` inside `DocuTutor_Engine/`.

## Repo notes

- Qdrant URL is hardcoded as `http://localhost:6333` in `indexingpipe.py`.
- `.active_collection` is runtime state and should not be committed.
- Use `.env.example` and don’t commit secrets.
