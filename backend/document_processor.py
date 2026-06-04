import os
import io
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_bytes(file_bytes: bytes, filename: str):
    """
    Extracts text from uploaded file bytes based on the filename extension.
    """
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension == '.txt':
        return file_bytes.decode("utf-8", errors="ignore")
    
    elif extension == '.pdf':
        pdf_stream = io.BytesIO(file_bytes)
        pdf_reader = PdfReader(pdf_stream)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    
    elif extension == '.docx':
        docx_stream = io.BytesIO(file_bytes)
        doc = Document(docx_stream)
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
