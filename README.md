# 📚 Math Question Generator

A local, LLM-powered web application built with Streamlit that automatically generates customized math questions from your study materials. Simply upload a document, select a Bloom's Taxonomy level, and let the AI generate high-quality, diverse, and well-structured mathematical problems for practice or assessment.

## ✨ Features

- **Document Processing:** Supports extracting text from `.pdf`, `.docx`, and `.txt` files.
- **Fast Concept Retrieval:** Uses TF-IDF and cosine similarity to quickly find the most relevant document chunks without needing a heavy embedding model or GPU.
- **Local AI Generation:** Integrates with local LLMs via [Ollama](https://ollama.com/) (e.g., `llama3`, `mistral`, `phi3`) for complete privacy and offline question generation.
- **Bloom's Taxonomy Levels:** Generates questions tailored to specific cognitive levels:
  - Remember
  - Understand
  - Apply
  - Analyze
  - Evaluate
  - Create
- **Quality Assured:** Prompts are structured to ensure 5 completely unique questions containing all necessary numerical values and contexts, specifically targeted as mathematical problems rather than theoretical concepts.

## 🚀 Setup and Installation

### Prerequisites

1. **Python 3.x** installed on your system.
2. **Ollama:** You must have [Ollama](https://ollama.com/) installed and running locally to serve the LLM.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yusufdhansay/mathquegenerator.git
   cd mathquegenerator
   ```

2. **Install Python dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the LLM model using Ollama:**
   By default, the application uses `llama3`. Run this in your terminal to download the model before starting the app:
   ```bash
   ollama pull llama3
   ```

### Running the Application

Start the Streamlit development server:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 📁 Project Structure

- **`app.py`**: The main Streamlit web application UI and orchestration logic.
- **`document_processor.py`**: Contains logic for extracting text from PDFs, Word docs, and text files, as well as chunking the text using LangChain's RecursiveCharacterTextSplitter.
- **`question_generator.py`**: Handles the TF-IDF chunk retrieval and interacts with the local Ollama LLM to generate the math questions based on customized Bloom's taxonomy prompts.
- **`requirements.txt`**: List of Python dependencies required to run the application.

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Web framework for data apps
- [LangChain](https://python.langchain.com/) - Text processing tools
- [Ollama](https://ollama.com/) - Local LLM runner
- [Scikit-Learn](https://scikit-learn.org/) - TF-IDF Vectorization for chunk retrieval
- [PyPDF2](https://pypi.org/project/PyPDF2/) / [python-docx](https://python-docx.readthedocs.io/) - File parsing
