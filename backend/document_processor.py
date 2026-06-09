import os
import io
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extracts text from uploaded file bytes based on the filename extension.
    Supports txt, pdf, and docx. Handles encoding fallbacks and table extractions.
    """
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension == '.txt':
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 encoding if UTF-8 fails
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file: {str(e)}")
    
    elif extension == '.pdf':
        try:
            pdf_stream = io.BytesIO(file_bytes)
            pdf_reader = PdfReader(pdf_stream)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except PdfReadError as pre:
            raise ValueError(f"Corrupted or encrypted PDF document: {str(pre)}")
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document: {str(e)}")
    
    elif extension == '.docx':
        try:
            docx_stream = io.BytesIO(file_bytes)
            doc = Document(docx_stream)
            
            # Extract paragraphs
            paragraphs_text = [para.text for para in doc.paragraphs]
            
            # Extract tables (ensures text inside Word tables isn't ignored)
            tables_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    filtered_row = [t.strip() for t in row_text if t and t.strip()]
                    if filtered_row:
                        tables_text.append(" | ".join(filtered_row))
            
            combined_text = "\n".join(paragraphs_text + tables_text)
            return combined_text
        except Exception as e:
            raise ValueError(f"Failed to parse Word (.docx) document: {str(e)}")
    
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
