# 🤖 Multimodal-AI-ChatBot-for-YouTube-Video-QA

## 📌 Overview

This project develops and evaluates a Retrieval-Augmented Generation (RAG) system for question answering on ServiceNow YouTube video content. We build a full pipeline for extracting transcripts, chunking text, embedding, retrieval, and answer generation—then benchmark multiple LLMs (including Mistral, Flan-T5, DistilGPT2, GPT-3.5-turbo) for answer quality using both automated metrics and LLM-as-a-judge evaluation.

## 🚀 Key Features

- **🎤 Audio & Transcript Extraction:** Automated downloading, transcription (via Whisper), and cleaning of YouTube video audio.
- **📚 Intelligent Chunking:** Splitting transcripts into context-aware text chunks for more effective retrieval.
- **🔎 RAG Pipeline:** End-to-end system combining vector search (FAISS/Pinecone), retrieval, and generative QA.
- **🤖 Multi-Model Evaluation:** Compare DistilGPT2, Flan-T5, Mistral-7B, GPT-3.5-turbo, and others.
- **📊 Metrics & Analysis:** F1, ROUGE-L, and LLM-in-the-loop scoring for robust QA evaluation.
- **📝 Ready-to-Use Jupyter Notebooks:** Each step modularized and reproducible.

## 🧩 Project Steps

### 1. Data Acquisition & Preprocessing

    #### 🎯 Objective
        Extract, transcribe, and clean audio from ServiceNow YouTube videos.
    #### Notebooks:
    - 01_metadata_with_transcripts.ipynb
    - 01b_audio_download_and_transcription.ipynb

#### Workflow:

    - Gather video metadata and links from YouTube.
    - Download audio and transcribe using OpenAI Whisper.
    - Clean and structure transcript data.
    - Save as chunked CSVs for embedding.

### 2. Chunking, Embedding, and Vector Storage

    #### 🎯 Objective
        Divide transcripts into semantic chunks and store embeddings for fast retrieval.

#### Workflow:

    - Chunk transcripts for better context and overlap.
    - Generate embeddings (OpenAI, Sentence Transformers, etc).
    - Store in FAISS or Pinecone for fast similarity search.

### 3. Retrieval-Augmented QA Pipeline

    #### 🎯 Objective
    Implement RAG to answer questions about videos using chunk retrieval and LLMs.

    #### Workflow:
    - Accept audio or text queries.
    - Retrieve relevant transcript chunks via vector similarity.
    - Pass context + query to various LLMs for answer generation.

### 4. Multi-Model Answer Generation & Evaluation

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

    - /audio/ – YouTube video audio files
        - /audio_files/ - Audio Files of Youtube Videos
        - /ServiceNow_Audio_Transcripts/ - Audio Files Transcriptions
    - /Data/ – Metadata,Transcript and cleaned chunks files
        - /SNOW_YT_Videos.csv/ - Main DataSet consists of 22 Youtube Video Links
        - /ServiceNow_Youtube_Metadata_Clean.csv/ - Metadata of Main Youtube Videos Dataset
        - /video_metadata_with_transcripts.csv/ - Transcripts from Metadata of Main Youtube Videos
        - /processed_transcripts.csv/ - Preprocessed Transcriptions
        - /processed_cleaned_chunks.csv/ - Cleaned Chunks from Preprocessed Transcriptions
    - /faiss_store/ - Embedded and vectorized files
        - /index.faiss/
        - /index.pkl/
    - /logs/ - chunk preview, project log and validation report files
        - /chunk_previews.txt/
        - /project_log.md/
        - /validation_report.txt/
    - /notebooks/ – All finalized major workflow and evaluation notebooks
        - /01_data_metadata_exploration.ipynb/ - Downloading Metadata from Youtube Videos
        - /02_data_preprocessing_transcript_chunk.ipynb/ - Data Preprocessing, Cleaning and Chunking
        - /3b_model_test_mistral7B.ipynb/ - Mistral Model Generation for Text prompt Engineering
        - /04_whisper_without_Agent.ipynb/ - Whisper Model Generation for Audio to Text Prompt Engineeering without LangChain Agents
        - /05_whisper_with_Agent_deploy.ipynb/ - Whisper Model Generation with LangChain Agents - **GRADIO DEPLOYMENT** Included
        - /06_rag_with_sources.ipynb/ - LLM OPenAI model to generate Answers with Source Snippets
        - /07_ QA_Calls_for_Evaluation.ipynb/ - Evaluations on all the Models used in this project
        - /08_deployment_gradio_test.ipynb/ - **GRADIO DEPLOYMENT without AGENT**
        - /09_matplot_evaluations.ipynb/ - Plots and Charts for this project
    - /Sample Models/ - All test Models
        - /03a_model_test_distilgpt2.ipynb/
        - /3c_model_test_flan-t5-base.ipynb/
        - /3d_model_test_flan-t5-large.ipynb/
    - /results/ – Model outputs, metrics, and evaluation CSVs
        - /all_model_eval_results.csv/ - Details from Evaluation notebook
        - /model_outputs.csv/- Details from each notebook individually
    - /requirements.txt/
    - /Multimodal AI Chatbot for YouTube Video QA.pptx/

## 🤝 Acknowledgements

    - ServiceNow (YouTube data)
    - Hugging Face, OpenAI, Together.ai, HuggingFace for open models and APIs
    - Community notebooks and repos for RAG and LLM evaluation
