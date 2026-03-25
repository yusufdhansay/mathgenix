from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from document_processor import get_text_chunks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Dictionary mapping Bloom's taxonomy levels to specific generation instructions
BLOOMS_INSTRUCTIONS = {
    "Remember": "Focus on fundamental mathematical facts, formulas, and identifying equations. Write problems like: State the formula for... What is the value of... Identify the property used in...",
    "Understand": "Focus on mathematical comprehension. Write problems that require interpreting a graph, translating a word problem into an equation, or explaining math concepts through examples. Example: Write an equation that represents the following situation...",
    "Apply": "Focus on strictly solving mathematical problems using given formulas or procedures. Provide numerical values or specific scenarios where the student must calculate an answer. Example: Solve for x in the equation... Calculate the area of...",
    "Analyze": "Focus on breaking down complex mathematical problems. Write problems that require students to find the error in a given solution, compare two different methods of solving an equation, or determine the relationship between mathematical variables.",
    "Evaluate": "Focus on mathematical justification and optimization. Write problems where students must evaluate whether a given mathematical solution is correct and justify why, or determine the most efficient method to solve a complex equation.",
    "Create": "Focus on mathematical modeling and synthesis. Write problems asking the student to construct a mathematical model (like an equation or function) that satisfies specific constraints, or design a word problem that results in a specific mathematical outcome."
}

def retrieve_relevant_chunks(chunks, query, k=5):
    """
    Uses TF-IDF + cosine similarity to find the most relevant text chunks for a query.
    This is extremely fast and uses no GPU/LLM — just simple math.
    """
    if not chunks:
        return []
    
    # Combine chunks with the query for TF-IDF vectorization
    all_texts = chunks + [query]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # The last vector is the query; compute similarity against all chunks
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Get indices of the top-k most similar chunks
    top_k = min(k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

def generate_questions(text: str, taxonomy_level: str, model_name: str = "llama3"):
    """
    Generates questions based on the provided text and Bloom's Taxonomy level using Ollama.
    """
    try:
        # Initialize the local Ollama LLM (NO embeddings model needed anymore!)
        llm = Ollama(model=model_name)
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is installed and running with the '{model_name}' model."

    # Retrieve specific instructions for the selected Bloom's level
    level_instruction = BLOOMS_INSTRUCTIONS.get(taxonomy_level, "Generate questions based on the text.")

    try:
        chunks = get_text_chunks(text)
        if not chunks:
            return "No valid text found in the document to generate questions."
        
        # Use TF-IDF to retrieve relevant chunks (instant, no LLM needed)
        query = f"Mathematical concepts, formulas, and problems suitable for {taxonomy_level} questions."
        relevant_chunks = retrieve_relevant_chunks(chunks, query, k=5)
        context_text = "\n\n".join(relevant_chunks)
    except Exception as e:
        return f"Error processing document context: {str(e)}"

    # Create the prompt template
    prompt_template = """
    You are an expert Math teacher and curriculum designer. Please analyze the following context and generate exactly 5 unique, highly diverse, and high-quality MATHEMATICS questions based on it.
    
    CRITICAL INSTRUCTIONS:
    1. The output MUST be actual math problems to solve (e.g., calculations, word problems), NOT theoretical, definitional, or historical questions.
    2. Topic Diversity (CRUCIAL): The provided context likely covers multiple different concepts or subtopics. You MUST extract different topics/formulas/concepts from the context and ensure each of the 5 questions is based on a completely different core concept from the context. DO NOT cluster questions around a single topic. Ask questions from different parts of the provided context.
    3. Complete Information: Every question MUST contain all the necessary numerical values, context, and data required to solve it. For example, if asking to calculate a final price with GST, you MUST provide the base price and the GST rate in the question itself. Never generate a question with missing information.
    4. Zero Repetition: Ensure each question is completely unique in structure, phrasing, and mathematical operation.
    
    The math problems must strictly target the following Bloom's Taxonomy level: "{taxonomy_level}". 
    {level_instruction}
    
    CONTEXT TO ANALYZE:
    {context}
    
    Please output ONLY the numbered list of exactly 5 math problems. Do not include answers or conversational filler.
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "taxonomy_level", "level_instruction"],
        template=prompt_template,
    )

    try:
        # Format the prompt and send to LLM
        formatted_prompt = prompt.format(
            context=context_text,
            taxonomy_level=taxonomy_level,
            level_instruction=level_instruction
        )
        
        response = llm.invoke(formatted_prompt)
        
        # Basic cleanup
        questions = response.strip()
        if not questions:
            return "The model returned an empty response."
        return questions
        
    except Exception as e:
        return f"Error generating questions: {str(e)}"
