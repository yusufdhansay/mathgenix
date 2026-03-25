import streamlit as st
import os

from document_processor import extract_text_from_file
from question_generator import generate_questions

st.set_page_config(
    page_title="Math Question Generator",
    page_icon="📚",
    layout="centered"
)

def main():
    st.title("📚 Math Question Generator")
    st.markdown("Generate math questions based on Bloom's Taxonomy from your uploaded study materials.")
    
    st.sidebar.header("Configuration")
    
    # Selection of Taxonomy Level
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    selected_level = st.sidebar.selectbox("Select Bloom's Taxonomy Level:", bloom_levels)
    
    # Optional model selection (if they have multiple ollama models pulled)
    st.sidebar.markdown("---")
    model_name = st.sidebar.text_input("Ollama Model Name", value="llama3", help="E.g., llama3, mistral, phi3")

    st.markdown("### 1. Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or DOCX file containing the math topic.",
        type=["pdf", "txt", "docx"]
    )
    
    st.markdown("### 2. Generate Questions")
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Analyzing document and generating questions..."):
                try:
                    # 1. Extract text
                    text = extract_text_from_file(uploaded_file)
                    
                    if not text.strip():
                        st.error("Could not extract any text from the document. Please try a different file.")
                        return

                    # 2. Generate questions
                    questions = generate_questions(text, selected_level, model_name=model_name)
                    
                    # 3. Display results
                    st.subheader(f"Generated Questions ({selected_level})")
                    st.markdown(questions)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Tip: If you're seeing connection errors, make sure you have installed the Ollama application and pulled the model using `ollama pull name_of_model` in your terminal.")

    else:
        st.info("Please upload a document to proceed.")

if __name__ == "__main__":
    main()
