# 📚 MathGenix: Advanced AI Math Assessment Hub

MathGenix is a professional, full-stack AI-powered mathematics assessment platform. It enables educators and students to upload study documents (PDF, Word, TXT), extract mathematical concepts, and dynamically generate high-quality math questions aligned with **Bloom's Taxonomy levels** complete with dynamic step-by-step LaTeX solution keys and automated symbolic validation.

Featuring a high-performance **FastAPI backend** and a premium **React + Vite** glassmorphism single-page frontend, MathGenix supports both **Local Ollama Inference** (offline mode) and **Groq Cloud LPU Inference** (high-speed cloud mode, generating 5 questions in ~3 seconds at 300+ tok/s).

---

## ✨ Core Features

1. **Dual Inference Engine (Cloud & Local):**
   * **Cloud Mode (Groq LPU):** Seamlessly routes prompts to Groq's high-speed chips (supporting `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, etc.) generating assessments instantly.
   * **Local Mode (Ollama):** Run completely offline on your local machine using models like `qwen2-math`, `deepseek-r1`, or `llama3`.
   
2. **Automated SymPy Verification Engine:**
   * Uses Python's symbolic mathematics library (**SymPy**) to mathematically verify the accuracy of the generated solutions.
   * Displays dynamic verification badges:
     * `✅ Verified`: Symbolic validation confirmed the answer matches the solution steps.
     * `⚠️ Unverified`: LLM output was mathematically sound but could not be parsed by SymPy.
     * `⏭️ Conceptual`: Non-numerical / conceptual problem (verification skipped).
     * `❌ Check Failed`: Validation failed (checks warning output for teacher review).

3. **High-Fidelity Printable Worksheet View:**
   * Designed with CSS print media rules to generate professional, clean worksheets via `window.print()` (Export as PDF).
   * Includes custom student worksheet spacing ("Show working here:") and a separate, toggleable **Answer Key / Solution Key** page containing KaTeX step-by-step derivations.
   * Allows exporting assessment decks in structured Markdown format.

4. **Robust LaTeX & Matrix Typesetting:**
   * Features a custom React `MathRenderer` leveraging KaTeX.
   * Employs backend regex mapping (`sanitize_latex` in `question_generator.py`) to format row breakers (`\\`) and LaTeX commands (`\lambda`, `\frac`) correctly, preventing JSON parser character-stripping bugs and ensuring compatibility with Python 3.13.

5. **Dynamic API Gateway Configuration UI:**
   * Built for seamless hosting on platforms like **Vercel** and **Render**.
   * Provides a dynamic API input panel in the **Server Status** (Settings) tab (persisted in `localStorage`) which resolves localhost mixed-content blocks and macOS IPv6 DNS conflicts (`localhost` vs `127.0.0.1`) without rebuilding/redeploying.

6. **Clear Extraction Diagnostics**:
   * Immediate file processing diagnostics displayed on card uploads. If a file fails to process, the system displays the error message inside the card instead of resetting the dropzone.

---

## 🚀 Setup & Installation

### Prerequisites
* **Python 3.10+** (with virtual environment support)
* **Node.js 18+** & **npm**
* **Ollama** (optional, for local offline generation)
* **Groq Cloud API Key** (free from [console.groq.com](https://console.groq.com))

---

## 🛠️ Running Locally

### 1. Configure the Backend Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
OLLAMA_API_URL=http://127.0.0.1:11434
```

### 2. Start the FastAPI Backend
```bash
# 1. Activate python virtual environment
source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Start uvicorn server on all interfaces (runs on port 8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Swagger API Docs can be inspected at: `http://127.0.0.1:8000/docs`*

### 3. Start the React Frontend
```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start Vite dev server (configured to bind to 127.0.0.1)
npm run dev
```
*The app is served at `http://127.0.0.1:5173/`.*

---

## 🌐 Public Cloud Hosting (Vercel + Render)

### Part 1: Deploy Backend (Render)
1. Push your repository to **GitHub**.
2. Create a new **Web Service** on Render pointing to your GitHub repository.
3. Configure settings:
   * **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Build Command**: `pip install -r requirements.txt`
4. Under **Advanced**, add the environment variable `GROQ_API_KEY` with your cloud key.
5. Deploy and copy your Render web service URL (e.g. `https://mathgenix-backend.onrender.com`).

### Part 2: Deploy Frontend (Vercel)
1. Go to Vercel, import your project.
2. Set **Root Directory** to `frontend`.
3. Set **Framework Preset** to `Vite`.
4. Click **Deploy**.
5. Once your Vercel frontend compiles and opens (e.g. `https://mathgenix.vercel.app`):
   * Navigate to the **Server Status** (Settings) tab.
   * Paste your Render backend URL into the **API Gateway Configuration** field.
   * Click **Save URL**.
   * The app will instantly connect and load your preset Groq API key automatically!

---

## 📂 Project Architecture

```
├── backend/
│   ├── main.py                 # FastAPI endpoints, CORS middleware, file uploads
│   ├── document_processor.py   # In-memory document stream reader (PDF, DOCX, TXT)
│   ├── question_generator.py   # Ollama prompt engine & TF-IDF relevance retriever
│   ├── groq_generator.py       # Groq OpenAI-compatible client & model details
│   └── answer_verifier.py      # Symbolic computation engine using SymPy
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── MathRenderer.jsx # Regex-based KaTeX compiler
│   │   ├── App.jsx             # Sidebar manager, Workspace view, settings panel
│   │   ├── App.css             # Glassmorphism dark mode system layout
│   │   └── main.jsx            # React mounting script
│   ├── package.json            # Node dependency configuration
│   └── vite.config.js          # Vite config
├── requirements.txt            # Python backend dependencies
└── README.md                   # System documentation
```
