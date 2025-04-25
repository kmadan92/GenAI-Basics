# 🤖 AI Assistant Chatbot

An intelligent assistant powered by **LangChain**, **OpenAI embeddings**, and **Qdrant**. This chatbot learns from your PDFs and answers any relevant questions based on the content.

---

## 🚀 Features

- Reads and understands PDF files using OpenAI Embeddings
- Uses Qdrant as a vector store for fast and scalable semantic search
- Built with LangChain for streamlined retrieval and generation
- Lightweight and easy to use — just drop in PDFs and run!

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. Install dependencies
    pip install -r requirements.txt

3. Prepare your environment

    Create a .env file using the provided env.sample

    Add your OpenAI API key:

    OPENAI_API_KEY=your-api-key-here

4. dd your PDFs

    Drop all your PDF documents into the /pdf folder.

5. Run the chatbot
    python main.py


## How It Works

- The assistant loads all PDFs from the /pdf folder.

- It creates embeddings using OpenAI’s API.

- These embeddings are stored in Qdrant for efficient retrieval.

- When a user asks a question, the assistant searches relevant chunks and responds using a language model.

![rag](https://github.com/user-attachments/assets/4ba943bb-d13e-455b-8134-0936c513fc3c)

📌 Requirements

Python 3.8+

OpenAI account/API key

Internet connection (to access OpenAI API)
