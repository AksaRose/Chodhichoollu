# Docling Chatbot 🤖💬

Welcome to the Docling Chatbot! This project is a super cool, document-aware chatbot that lets you upload PDFs and then chat with them. Ever wanted to ask your lengthy research papers questions and get instant answers? Now you can! We've spiced things up with formula, picture, and code enrichment to make your document interactions even smarter.

## ✨ Features

*   **PDF Upload & Chat**: Seamlessly upload your PDF documents and start a conversation.
*   **Intelligent Responses**: Powered by a LangGraph agent, the chatbot uses the uploaded document to answer your questions.
*   **Formula-Aware**: Understands and processes mathematical formulas within your documents.
*   **Picture Description**: Can describe pictures found in your PDFs (thanks to `smolvlm_picture_description`!).
*   **Code Enrichment**: Recognizes and handles code snippets in your documents.
*   **Interactive UI**: A sleek React frontend for a smooth chat experience.

## 🚀 Tech Stack

This project is a harmonious blend of Python and TypeScript, making for a powerful and flexible system!

### Backend (Python)

*   **FastAPI**: For building the robust and speedy API.
*   **LangGraph**: The brain of our chatbot, orchestrating the conversational flow.
*   **LangChain**: Tools for interacting with language models and external data.
*   **Docling**: For advanced PDF parsing, chunking, and enrichment (formulas, pictures, code!).
*   **Milvus**: Our vector store of choice for super-fast similarity searches.
*   **HuggingFace Embeddings**: For turning text into smart vectors.
*   **`python-dotenv`**: Keeping our secrets safe and sound (environment variables).

### Frontend (React with TypeScript)

*   **React**: The magical JavaScript library for building dynamic user interfaces.
*   **TypeScript**: Adding types to JavaScript for more robust and maintainable code.
*   **Vite**: A blazing-fast build tool for modern web projects.
*   **`react-markdown`**, **`remark-math`**, **`rehype-katex`**: For beautifully rendering markdown and LaTeX formulas in the chat.

## 🏃‍♀️ How to Run Locally (Get Your Bot Buddy Going!)

Follow these steps to get your very own Docling Chatbot up and running on your machine.

### Prerequisites

Before you start, make sure you have these installed:

*   **Python 3.8+**
*   **Node.js & npm** (or yarn)

### 1. Clone the Repository

First things first, grab the code!

```bash
git clone https://github.com/your-repo/docling-chatbot.git # Replace with actual repo URL
cd docling-chatbot
```

### 2. Backend Setup

Let's get the Python server ready.

```bash
# Create and activate a virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables (if any)
# Create a .env file in the root directory and add necessary variables.
# Example:
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your_langsmith_key
# GOOGLE_API_KEY=your_google_api_key
```

Once dependencies are installed, you can fire up the backend server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

This will start the FastAPI server, usually accessible at `http://localhost:8001`.

### 3. Frontend Setup

Now for the pretty face of our chatbot!

```bash
cd frontend

# Install Node.js dependencies
npm install # or yarn install
```

And then, launch the frontend development server:

```bash
npm run dev # or yarn dev
```

This will typically open the React app in your browser at `http://localhost:5173`.

## 🤖 How to Use

1.  **Upload a PDF**: On the left side of the application, you'll see an "Upload PDF" section. Click the file input, select your PDF, and hit "Upload PDF".
2.  **Wait for Processing**: The backend will process your PDF, chunking it and embedding its content into our vector store. You'll see a success message once it's done.
3.  **Start Chatting!**: Once your PDF is uploaded, head over to the chat interface on the right. Type your questions related to the document in the input box and press Enter or click "Send".
4.  **Get Smart Answers**: The chatbot will use the uploaded document's content to provide informed and context-aware responses, even handling complex formulas and code snippets!

Have fun chatting with your documents! If you find any bugs or have cool ideas, feel free to contribute.
