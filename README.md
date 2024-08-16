# Vecky 🚀

Vecky is a lightweight vector database implementation that uses PostgreSQL with the pgvector extension for persisting embeddings and supporting multi-document operations.

## Checkout the blog posts: 
   - [Demystifying Vector Stores: A Beginners Guide for Improved RAG Systems using Numpy](https://medium.com/@gurramakhileshwar333/demystifying-vector-stores-a-beginners-guide-for-improved-rag-systems-using-numpy-a9bd9eba0142)
   - [Demystifying Vector Stores: A Beginners Guide for Improved RAG Systems using PostgreSQL](https://medium.com/@gurramakhileshwar333/demystifying-vector-stores-a-beginners-guide-for-improved-rag-systems-using-postgresql-d191cdd4d386)


## Features 🌟
- 🐘 PostgreSQL backend with pgvector extension for efficient vector operations
- 🧠 Support for multiple embedding models:
  - Sentence Transformer
  - OpenAI Embedding Model
- 🔍 Similarity search metrics:
  - Euclidean distance
  - Cosine similarity
- 📚 Multi-document support
- 🗂️ Collection-based document organization
- 🛠️ Flexible API for adding documents and performing searches
- APIs for reading stats about collections and setting similarity metric

## Prerequisites 📋

- Docker
- Python 
- PostgreSQL with pgvector extension

## Quick Start 

1. Start a PostgreSQL instance with pgvector:

   ```bash
   docker run --name pgvector-demo -e POSTGRES_PASSWORD=password -p 5433:5432 -d ankane/pgvector
   
2. Set up your OpenAI API key as an environment variable (if you choose to use openai's embedding model):
   ```bash
      export OPENAI_API_KEY=your_api_key_here
   
3. Install requirements:
   ```bash
      poetry install

4. Run the main script:
   ```bash
      python main.py
