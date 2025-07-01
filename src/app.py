from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.docstore.document import Document
import gradio as gr
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

documents = [Document(page_content="ServiceNow uses AI for incident classification.", metadata={"source": "intro.txt"})]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_store = FAISS.from_documents(documents, embedding=embedding_model)

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_store.as_retriever(),
    return_source_documents=True
)

def answer_question(question):
    result = qa_chain(question)
    answer = result["result"]
    sources = []
    for doc in result["source_documents"]:
        snippet = doc.page_content[:200].strip()
        sources.append(f"📄 {doc.metadata['source']}
🔎 {snippet}")
    return answer, "\n\n".join(sources)

gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="Ask a ServiceNow question..."),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="ServiceNow QA Bot",
    description="Ask questions about ServiceNow based on transcripts."
).launch()
