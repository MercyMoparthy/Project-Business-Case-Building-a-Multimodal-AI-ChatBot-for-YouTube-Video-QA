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
    Notebooks:
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
    - Whisper without Agents
    - Whisper with Agents

#### Evaluation Metrics:
    - Token-level F1 (exact word overlap)
    - LLM-as-a-Judge (use GPT-3.5/4 to rate factual correctness and completeness)
    - Manual inspection for qualitative insights
    - Rouge, Exact match and Bleu for individual models

#### 🛠️ Tech Stack
* **Programming Language:**
  * Python 3.x
* **Core Libraries & Frameworks:**
  * **pandas, numpy** — Data manipulation and analysis
  * **scikit-learn** — Machine learning utilities and evaluation
  * **matplotlib** — Data visualization
  * **Jupyter Notebooks** — Interactive development and documentation
* **Large Language Models & Frameworks:**
  * **Hugging Face Transformers:** - Pretrained and finetuned models (e.g., DistilGPT2, Flan-T5)
  * **OpenAI GPT API** — LLM-based generation and QA
  * **Together.ai API** — For alternative LLM endpoints
  * **Audio Processing:** - Whisper (OpenAI Whisper ASR via Hugging Face/transformers) — Automatic speech recognition for audio-to-text transcription
* **Retrieval-Augmented Generation (RAG):**
  * **LangChain** — Building and orchestrating RAG pipelines
  * **LangSmith** — Workflow tracking, evaluation, and dashboarding
* **Embeddings & Vector Stores:**
  * **Sentence Transformers or Hugging Face Embedding models** — Text embedding
  * **FAISS** — Local vector database for similarity search
  * **Pinecone** — Managed vector database (optional/integrated for scalability)
* **APIs & Integrations:**
  * **OpenAI, Hugging Face, Together.ai APIs** — Model inference, embedding, and transcription services
* **Development Environments:**
  * **VS Code** — Code development
  * **Google Colab** — Cloud-based notebook execution and prototyping

#### 📈 Results & Insights
    - Chunk-based retrieval dramatically improves LLM QA accuracy over raw transcripts.
    - Flan-T5 models achieved highest F1/ROUGE on strict metrics, but Mistral-7B delivered the most robust, human-like answers (per manual and LLM-based review).
    - Automated metrics can undervalue models that paraphrase or elaborate; LLM-in-the-loop and human analysis are essential for fair evaluation.

## 🔍 Observations

    - F1 and ROUGE are useful but insufficient for open-ended QA evaluation; they miss high-quality paraphrasing and extra context.
    - Mistral-7B’s answers often scored lower on F1 but excelled in factuality and completeness when checked by LLMs or humans.
    - Whisper model with Agent excelled in factuality and completeness compared to without Agent
    - Multi-metric and LLM-as-a-judge evaluation is recommended for any serious RAG QA benchmark.

## 📂 Repository Structure
        Project-Business-Case-Building-a-Multimodal-AI-ChatBot-for-YouTube-Video-QA/
        │ audio                                            # Audio-related resources
        │   ├── audio_files                                # Raw YouTube video audio files
        │   └── ServiceNow_Audio_Transcripts               # Transcriptions of audio files
        │
        │ Data                                             # Data and processed files
        │   ├── SNOW_YT_Videos.csv                         # Main dataset: 22 YouTube video links
        │   ├── ServiceNow_Youtube_Metadata_Clean.csv      # Cleaned metadata for YouTube videos
        │   ├── video_metadata_with_transcripts.csv        # Combined metadata and transcripts
        │   ├── processed_transcripts.csv                  # Preprocessed transcriptions
        │   └── processed_cleaned_chunks.csv               # Cleaned transcript chunks
        │
        │ faiss_store                                      # FAISS vector store
        │   ├── index.faiss
        │   └── index.pkl
        │
        │ logs                                             # Logs and validation reports
        │  ├── chunk_previews.txt
        │  ├── project_log.md
        │  └── validation_report.txt
        │
        │ notebooks                                        # Finalized workflow notebooks
        │  ├── 01_data_metadata_exploration.ipynb
        │  ├── 02_data_preprocessing_transcript_chunk.ipynb
        │  ├── 03b_model_test_mistral7B.ipynb
        │  ├── 04_whisper_without_Agent.ipynb             
        │  ├── 05_whisper_with_Agent_deploy.ipynb          # DEPLOYMENT with AGENT
        │  ├── 06_rag_with_sources.ipynb
        │  ├── 07_QA_Calls_for_Evaluation.ipynb
        │  ├── 08_deployment_gradio_test.ipynb             # DEPLOYMENT without AGENT
        │  └── 09_matplot_evaluations.ipynb
        │
        │ Sample_Models                                    # Prototype/test model notebooks
        │  ├── 03a_model_test_distilgpt2.ipynb
        │  ├── 03c_model_test_flan-t5-base.ipynb
        │  └── 03d_model_test_flan-t5-large.ipynb
        │
        │/results/                                         # Model outputs and evaluation CSVs
        │  ├── all_model_eval_results.csv
        │  └── model_outputs.csv
        │
        │ requirements.txt                                 # Project dependencies
        │ Multimodal AI Chatbot for YouTube Video QA.pptx  # Project presentation


## 🤝 Acknowledgements

* **ServiceNow** — For providing the valuable YouTube video content and data that formed the backbone of this project.
* **Hugging Face, OpenAI, and Together.ai** — For access to powerful open-source language models, APIs, and supporting tools that enabled advanced natural language understanding and generation.
* **Hugging Face Transformers & Datasets** — For the libraries and resources that accelerated the development and experimentation of model architectures and pipelines.
* **LangChain & LangSmith** — For their modular frameworks and insightful dashboards that streamlined the implementation and evaluation of retrieval-augmented generation (RAG) workflow
* **Google Colab & Visual Studio Code (VS Code)** — For flexible, collaborative, and scalable development environments throughout all project phases.
* **Community Resources & Open Notebooks** — For inspiration, reusable code snippets, and best practices shared by the open-source AI and ML community, especially in RAG, LLM evaluation, and multimodal workflows.
