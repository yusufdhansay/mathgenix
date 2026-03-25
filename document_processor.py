import os
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_file(uploaded_file):
    """
    Extracts text from an uploaded file based on its extension.
    """
    filename = uploaded_file.name
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension == '.txt':
        return str(uploaded_file.read(), "utf-8")
    
    elif extension == '.pdf':
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text
    
    elif extension == '.docx':
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits a large text string into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

