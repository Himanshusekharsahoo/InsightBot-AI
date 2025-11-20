# 🧠 InsightBot AI  
**Private AI-Powered Search & Q&A System for Teams**

InsightBot AI is an intelligent, privacy-first knowledge search assistant that lets teams ask natural language questions over their own documents. It uses **Retrieval-Augmented Generation (RAG)** to find, reason over, and cite answers from organizational data sources — like PDFs, Google Drive files, or GitHub repositories — all within a secure and modular architecture.

---

## 📘 Overview

InsightBot AI empowers teams to unlock knowledge buried in their documents by combining **semantic search**, **vector embeddings**, and **LLM reasoning**. It features a clean FastAPI backend, a modern React-based frontend, and full Dockerized deployment for easy setup.

Whether you’re building an internal document search system, an enterprise chatbot, or an AI assistant for your data, InsightBot AI serves as a robust foundation to demonstrate end-to-end backend + AI integration skills.

---

## 🚀 Features

- 🔍 **Context-Aware Q&A** – Combines semantic retrieval with large language model reasoning  
- 🧩 **Multi-Source ETL Pipeline** – Automatically ingests and indexes content from text, PDF, or web sources  
- 🧠 **Vector Search with Embeddings** – Utilizes Sentence-Transformers for high-quality semantic similarity  
- 🔐 **Secure Authentication** – Implements JWT-based user authentication and access control for APIs  
- 🧱 **Modular Architecture** – Independent backend, data connectors, and model service layers  
- ☁️ **Dockerized Deployment** – One-click local or production setup with Docker Compose  
- 🧰 **Extensible LLM Interface** – Switch easily between OpenAI, Gemini, or local Hugging Face models  

---

## 🏗️ Tech Stack

| Category   | Technologies                                                                 |
|------------|------------------------------------------------------------------------------|
| Backend    | FastAPI · SQLAlchemy · LangChain · Pydantic · PostgreSQL                    |
| AI/ML      | Hugging Face Transformers · Sentence-Transformers · Vespa/DeepLake (Vector DB) |
| Frontend   | Next.js · React · Tailwind CSS                                               |
| DevOps     | Docker · Poetry · GitHub Actions                                             |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Himanshusekharsahoo/InsightBot-AI.git
cd InsightBot-AI
```

### 2. Environment setup
Create a `.env` file in the root directory and configure your keys and settings:
```env
OPENAI_API_KEY=your_key_here
POSTGRES_USER=insight_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=insightbot_db
```

### 3. Run with Docker Compose
```bash
docker-compose up --build
```

Once the build completes successfully, access the API documentation here:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧩 Usage

1. **Upload Documents**  
   Use the `/upload` endpoint or the web UI to upload documents (`.pdf`, `.txt`, `.md`, etc.)

2. **Ask Questions**  
   Query via the `/ask` endpoint or through the interactive chat interface

3. **Get Cited Answers**  
   InsightBot retrieves relevant segments, applies LLM reasoning, and delivers concise answers with citations

#### Example
**Q:** “How does the company define its remote work policy?”  
**A:** Found in Employee Handbook.pdf: employees may work remotely up to 3 days per week, subject to manager approval and team coordination.

---

## 🔍 Architecture Overview

```
                 ┌───────────────────────┐
                 │      Frontend (UI)    │
                 │ Next.js + Tailwind CSS│
                 └──────────┬────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │          Backend (FastAPI)            │
        │ Auth · Upload · Query · Admin APIs    │
        └──────────┬────────────────────────────┘
                   │
   ┌───────────────▼───────────────────────┐
   │        Vector Search & LLM Service    │
   │ Sentence-Transformers · LangChain     │
   └───────────────┬───────────────────────┘
                   │
        ┌──────────▼───────────────┐
        │   PostgreSQL + VectorDB  │
        │   Metadata & embeddings  │
        └──────────────────────────┘
```

---

## 🧰 Extensibility

- **Swap Models:** Easily plug in Hugging Face, Gemini, or OpenAI endpoints by updating configuration variables  
- **Add Data Sources:** Extend connectors to fetch data from Google Drive, Confluence, or Notion  
- **Frontend Rebranding:** Customize the UI theme or integrate OAuth for organizational branding  

---

## 🧠 Typical Workflow

1. User uploads internal documents  
2. ETL pipeline extracts clean text and stores it in the vector database  
3. Query request triggers semantic retrieval based on embeddings  
4. Retrieved chunks are provided as context to the chosen LLM  
5. The LLM composes a human-readable, cited response  

---

## 💡 Why InsightBot AI Stands Out

- Full-stack demonstration of AI + Backend + DevOps integration  
- Clean modular architecture suitable for production adaptation  
- End-to-end implementation of RAG pipeline with authentication and UI  
- Ideal for showcasing **AI engineering**, **backend development**, and **data systems** knowledge  

---

## 🧑‍💻 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

```bash
git checkout -b feature/your-feature
git commit -m "Add new data connector"
git push origin feature/your-feature
```

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and build on top of it.
