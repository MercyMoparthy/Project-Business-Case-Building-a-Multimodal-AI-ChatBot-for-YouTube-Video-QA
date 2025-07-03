# 🤖 Multimodal-AI-ChatBot-for-YouTube-Video-QA

## 📌 Overview

This project develops and evaluates a Retrieval-Augmented Generation (RAG) system for question answering on ServiceNow YouTube video content. We build a full pipeline for extracting transcripts, chunking text, embedding, retrieval, and answer generation—then benchmark multiple LLMs (including Mistral, Flan-T5, DistilGPT2, GPT-3.5-turbo) for answer quality using both automated metrics and LLM-as-a-judge evaluation.

## 🚀 Key Features

- **🎤 Audio & Transcript Extraction: Automated downloading, transcription (via Whisper), and cleaning of YouTube video audio.
- **📚 Intelligent Chunking: Splitting transcripts into context-aware text chunks for more effective retrieval.
- **🔎 RAG Pipeline: End-to-end system combining vector search (FAISS/Pinecone), retrieval, and generative QA.
- **🤖 Multi-Model Evaluation: Compare DistilGPT2, Flan-T5, Mistral-7B, GPT-3.5-turbo, and others.
- **📊 Metrics & Analysis: F1, ROUGE-L, and LLM-in-the-loop scoring for robust QA evaluation.
- **📝 Ready-to-Use Jupyter Notebooks: Each step modularized and reproducible.

## 🧩 Project Steps

### 1.  Data Acquisition & Preprocessing
    #### 🎯 Objective
        Extract, transcribe, and clean audio from ServiceNow YouTube videos.
    Notebooks:

    - 01_metadata_with_transcripts.ipynb
    - 01b_audio_download_and_transcription.ipynb

   ####  Workflow:
    - Gather video metadata and links from YouTube.
    - Download audio and transcribe using OpenAI Whisper.
    - Clean and structure transcript data.
    - Save as chunked CSVs for embedding.

### 2.  Chunking, Embedding, and Vector Storage
    #### 🎯 Objective
        Divide transcripts into semantic chunks and store embeddings for fast retrieval.

   #### Workflow:
    - Chunk transcripts for better context and overlap.
    - Generate embeddings (OpenAI, Sentence Transformers, etc).
    - Store in FAISS or Pinecone for fast similarity search.

### 3.  Retrieval-Augmented QA Pipeline
    #### 🎯 Objective
    Implement RAG to answer questions about videos using chunk retrieval and LLMs.

    #### Workflow:
    - Accept audio or text queries.
    - Retrieve relevant transcript chunks via vector similarity.
    - Pass context + query to various LLMs for answer generation.

### 4.  Multi-Model Answer Generation & Evaluation
    #### 🎯 Objective
        Benchmark different LLMs for QA answer quality.

    #### LLMs Benchmarked:
    - DistilGPT2
    - Flan-T5-base
    - Flan-T5-large
    - Mistral-7B-Instruct (via Together.ai and HuggingFace)
    - GPT-3.5-turbo (via OpenAI API)

    #### Evaluation Metrics:
    - Token-level F1 (exact word overlap)
    - LLM-as-a-Judge (use GPT-3.5/4 to rate factual correctness and completeness)
    - Manual inspection for qualitative insights

    #### 🛠️ Tech Stack
    - Python (pandas, numpy, sklearn, matplotlib)
    - Hugging Face Transformers (DistilGPT2, Flan-T5, etc.)
    - Whisper (for audio transcription)
    - FAISS / Pinecone (vector database)
    - OpenAI, HuggingFace & Together.ai APIs
    - Jupyter Notebooks

    #### 📈 Results & Insights
    - Chunk-based retrieval dramatically improves LLM QA accuracy over raw transcripts.
    - Flan-T5 models achieved highest F1/ROUGE on strict metrics, but Mistral-7B delivered the most robust, human-like answers (per manual and LLM-based review).
    - Automated metrics can undervalue models that paraphrase or elaborate; LLM-in-the-loop and human analysis are essential for fair evaluation.
    
## 🔍 Observations
    - F1 and ROUGE are useful but insufficient for open-ended QA evaluation; they miss high-quality paraphrasing and extra context.
    - Mistral-7B’s answers often scored lower on F1 but excelled in factuality and completeness when checked by LLMs or humans.
    - Multi-metric and LLM-as-a-judge evaluation is recommended for any serious RAG QA benchmark.

## 📂 File Structure
    - /audio_files/ – YouTube video audio files
    - /servicenow_audio_transcripts/ – Transcript files
    - /notebooks/ – All major workflow and evaluation notebooks
    - /results/ – Model outputs, metrics, and evaluation CSVs

## 🤝 Acknowledgements
    - ServiceNow (YouTube data)
    - Hugging Face, OpenAI, Together.ai for open models and APIs
    - Community notebooks and repos for RAG and LLM evaluation
