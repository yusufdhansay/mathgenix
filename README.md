# 📚 Math Question Generator & Quiz Arena (Phase 2)

Welcome to the **Math Question Generator & Quiz Arena**, a professional full-stack web application designed for interactive learning and automated math assessment. Built on a modular, high-performance architecture featuring a **FastAPI backend** and a premium **React + Vite** single-page application.

The platform extracts concept structures from academic textbooks (PDF, Word, TXT), dynamically locates core topics using local TF-IDF math models, and utilizes local LLMs (via **Ollama**) to generate high-fidelity math assessments across **Bloom's Taxonomy levels** complete with dynamic step-by-step LaTeX solution roadmaps.

---

## ✨ Full-Stack Features

1. **Generation & Accuracy Hub:**
   * **Structured JSON Outputs:** Robust parsing pipeline guarantees the LLM strictly returns questions, answers, and solutions.
   * **Step-by-Step solutions:** Auto-generated derivation blueprints for every problem, aiding students and teachers.
   * **KaTeX Inline & Block Math:** Fast, high-contrast, beautiful typesetting of formulas ($f(x) = \int x\,dx$) in real-time.

2. **Interactive Quiz Arena:**
   * Practice calculations directly inside the web platform.
   * Input answers, get instant grading feedback, and expand step-by-step solutions to inspect derivations.
   * Interactive score reports detailing overall performance and accuracy.

3. **Workspace Management:**
   * **Decks Directory:** Retrieve previous question sets from session storage and reload them instantly.
   * **Dynamic Model Detection:** Auto-discovers pulled Ollama models (e.g. `llama3`, `mistral`, `phi3`) from your local daemon.
   * **Markdown Exports:** Download print-ready question sheets with one click.

---

## 🚀 Setup & Installation

### Prerequisites

1. **Python 3.10+** (with virtual environment management).
2. **Node.js 18+** & **npm** (for compiling the React app).
3. **Ollama:** Download and run [Ollama](https://ollama.com/) locally to serve the LLMs.

---

## 🛠️ Running the Platform

To run the application, you will start the Backend API Gateway and the Frontend Dev Server concurrently.

### 1. Start the FastAPI Backend

1. Navigate to the project root directory.
2. Activate your virtual environment and install Python dependencies:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Boot the API Server using Uvicorn:
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   * *The server will begin running on `http://127.0.0.1:8000`. You can inspect the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.*

### 2. Start the React Frontend

1. Open a new terminal window and navigate to the `/frontend` directory:
   ```bash
   cd frontend
   ```
2. Install package dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   * *Open the local browser portal, typically served at `http://localhost:5173`.*

---

## 📂 Project Architecture

```
├── backend/
│   ├── main.py                 # FastAPI routing, middleware, and upload controller
│   ├── document_processor.py   # Extracts file streams & chunks text recursives
│   └── question_generator.py   # TF-IDF cosine ranking & Ollama prompt engine
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── MathRenderer.jsx # Regular expression LaTeX compiler using KaTeX
│   │   ├── App.jsx             # Sidebar manager, Workspace view, Quiz Arena
│   │   ├── index.css           # Premium glassmorphism dark system styling
│   │   └── main.jsx            # Vite DOM React mounting script
│   └── package.json            # Node.js project requirements
├── requirements.txt            # Unified Python dependency tracking manifest
└── README.md                   # Setup and system user documentation
```

---

## 📦 Tech Stack

*   **Backend:** FastAPI, Uvicorn, LangChain, PyPDF2, python-docx, scikit-learn
*   **Frontend:** React, Vite, KaTeX, Lucide Icons, Vanilla CSS
*   **AI Engine:** Local Ollama daemon running Llama-3, Mistral, or Phi-3
