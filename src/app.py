import os
from typing import Tuple, Union
import gradio as gr
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from transformers import pipeline
from google.colab import userdata # Assuming you'll run this part in Colab to get the key

# --- Configuration ---
# You might need to securely load this in a production app (e.g., using environment variables)
# For Colab, you can get it from userdata secrets
openai_api_key = userdata.get("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set. Please add it to Colab secrets.")

# Define the path to your saved FAISS index
faiss_store = "/content/drive/MyDrive/faiss_store"
if not Path(faiss_store).exists():
     raise FileNotFoundError(f"FAISS index not found at {faiss_store}. Please ensure it's saved there.")

# --- Model and Chain Initialization ---

# Load the speech-to-text model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Initialize the embedding model (used for loading FAISS)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the FAISS index from the local directory
faiss_index = FAISS.load_local(
    faiss_store,
    embedding_model,
    allow_dangerous_deserialization=True # Be cautious with untrusted sources
)

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Create the RetrievalQA chain
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- Gradio Interface Function ---

def answer_question(text_input: str, audio_input: Union[str, dict, None]) -> Tuple[str, str]:
    """
    Runs the RetrievalQA chain and formats the answer along with source snippets.
    Handles both text and audio input.

    Args:
        text_input: The string input from the text box.
        audio_input: The audio input, expected as a string filepath from Gradio.

    Returns:
      - answer text
      - formatted source list (one entry per document)
    """
    try:
        question = ""

        # Process audio input if provided (handle as a string filepath)
        if isinstance(audio_input, str) and audio_input:
             audio_filepath = audio_input
             try:
                 transcription_result = transcriber(audio_filepath)
                 question = transcription_result.get("text", "")
                 if not question:
                     return "Could not transcribe audio. Please try again.", ""
                 question = question.strip() # Remove leading/trailing whitespace from transcription
                 if not question:
                     return "Transcribed audio was empty. Please try again.", ""
             except Exception as trans_e:
                 return f"Error during transcription: {trans_e}", ""


        # Use text input if audio was not provided or transcription failed
        if not question and isinstance(text_input, str) and text_input.strip():
            question = text_input.strip()
            if not question:
                return "Please enter a question or record audio.", ""

        # Ensure there is a question to process after handling input types
        if not question:
             return "No valid question provided via text or audio.", ""

        # Run the QA chain with the processed question
        try:
            response = qa_chain(question)
            answer = response.get("result", "") # Use .get() with a default value

            # Build source list with preview snippets
            sources = []
            source_documents = response.get("source_documents", []) # Use .get() with a default empty list
            for doc in source_documents:
                text = doc.page_content.strip()
                preview = text[:400] + "..." if len(text) > 400 else text
                sources.append(f"📄 {doc.metadata.get('source', 'unknown')}
🔎 {preview}")

            if not answer:
                 # It's possible the QA chain returned successfully but with no relevant answer
                 return "Could not find a relevant answer in the documents.", "

".join(sources)


            return answer, "

".join(sources)

        except Exception as qa_e:
            return f"Error processing question with QA chain: {qa_e}", ""


    except Exception as e:
        err = f"An unexpected error occurred in answer_question: {e}"
        # Return the same error in both slots to display in UI
        return err, err

# --- Gradio Interface Definition ---

gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ask a ServiceNow question via text...", label="Text Input"), # Text input
        gr.Audio(sources=["microphone"], type="filepath", label="Ask your question via microphone") # Audio input
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],
    title="ServiceNow QA Assistant",
    description="RAG-powered assistant over your ServiceNow YouTube transcripts with text and speech input.",
    allow_flagging="never" # Use flagging_mode instead in newer Gradio versions
).launch()
