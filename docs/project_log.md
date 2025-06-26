## Data Collection and Preprocessing
- Loaded YouTube video from `data/SNOW_YT_Videos.csv`.
- Loaded YouTube video metadata to `data/ServiceNow_Youtube_Metadata_Clean.csv`.
- Loaded YouTube video transcripts to `data/video_metadata_with_transcripts.csv`.
- Preprocessed transcripts with NLTK lemmatization and LangChain text splitting (chunk_size=500, overlap=50).
- Processed 22 videos, generating 328 chunks.
- Saved processed data to `data/processed_transcripts.csv`.
- Challenges: Resolved KeyError by standardizing column names (e.g., Number to video_id).
- Validation report saved to `docs/validation_report.txt`.
- Chunk Preview Data saved to `docs/chunk_preview.csv`.
  Average: 966.9
  Minimum: 53
  Maximum: 1000
## Data Collection and Preprocessing
- Loaded YouTube video from `data/SNOW_YT_Videos.csv`.
- Loaded YouTube video metadata to `data/ServiceNow_Youtube_Metadata_Clean.csv`.
- Loaded YouTube video transcripts to `data/video_metadata_with_transcripts.csv`.
- Preprocessed transcripts with NLTK lemmatization and LangChain text splitting (chunk_size=500, overlap=50).
- Processed 22 videos, generating 328 chunks.
- Saved processed data to `data/processed_transcripts.csv`.
- Challenges: Resolved KeyError by standardizing column names (e.g., Number to video_id).
- Validation report saved to `docs/validation_report.txt`.
- Chunk Preview Data saved to `docs/chunk_preview.csv`.
  Average: 966.9
  Minimum: 53
  Maximum: 1000
