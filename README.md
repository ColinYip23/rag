# Chat with Multiple PDFs

A Streamlit Retrieval-Augmented Generation (RAG) application that lets users ask conversational questions about one or more uploaded PDF documents.

The app extracts text from PDFs, splits the text into chunks, creates embeddings using `hkunlp/instructor-xl`, stores the embeddings in a FAISS vector database, and uses a hosted Hugging Face chat model to generate answers based on the retrieved document context.

## Features

* Upload and process multiple PDF files
* Ask natural-language questions about uploaded documents
* Retrieve relevant document chunks using semantic search
* Generate answers using a hosted Hugging Face language model
* Maintain chat history during the current Streamlit session
* Run document embedding and FAISS vector search locally
* Display custom user and assistant chat messages

## How It Works

```text
PDF Uploads
    |
    v
Text Extraction using PyPDF2
    |
    v
Text Chunking
    |
    v
INSTRUCTOR-XL Embeddings
    |
    v
FAISS Vector Store
    |
    v
Retriever selects relevant chunks
    |
    v
Hugging Face Chat Model generates answer
    |
    v
Answer displayed in Streamlit
```

The app does not currently persist uploaded files, vector indexes, or chat history. Everything exists only during the active Streamlit session.

## Project Structure

```text
.
├── app.py              # Main Streamlit app and RAG pipeline
├── htmlTemplates.py    # Custom chat message HTML and CSS
├── requirements.txt    # Python dependencies
├── .env                # Local API key configuration
└── README.md
```

## Prerequisites

Before running the app, make sure you have:

* Python 3.9 or later
* `pip`
* A Python virtual environment
* A Hugging Face access token
* Internet access
* Enough RAM or VRAM to run the embedding model
* CUDA-enabled GPU if the app is configured to use `device="cuda"`

> Note: If your app uses `model_kwargs={"device": "cuda"}`, CUDA must be available. Otherwise, change the device to `"cpu"`.

## Installation

### 1. Clone or download the project

```bash
git clone <repository-url>
cd rag
```

If the project is already on your computer, open a terminal in the project folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If some packages are missing, install the main packages manually:

```bash
python -m pip install streamlit PyPDF2 python-dotenv faiss-cpu
python -m pip install langchain-community langchain-classic langchain-openai langchain-text-splitters
python -m pip install InstructorEmbedding sentence-transformers
```

If you are using PyTorch with CUDA, install the correct PyTorch version for your system from the official PyTorch installation page.

To check whether CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints:

```text
True
```

CUDA is available.

If it prints:

```text
False
```

use CPU mode or reinstall the correct CUDA-enabled PyTorch version.

## Configuration

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Do not commit your `.env` file to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.streamlit/
```

## Running the App

With the virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser.

If it does not open automatically, copy the local URL from the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit app.
2. Upload one or more PDF files from the sidebar.
3. Click **Process**.
4. Wait for the PDFs to be extracted, chunked, embedded, and stored in FAISS.
5. Type a question in the input box.
6. Press Enter.
7. Continue asking follow-up questions.

## Example Questions

```text
Summarize this document.
```

```text
What are the main points in the uploaded PDFs?
```

```text
Compare the conclusions from the documents.
```

```text
What does the document say about data preprocessing?
```

```text
Explain this section in simple terms.
```

```text
List the risks mentioned in the document.
```

## Running Without CUDA

If your computer does not have CUDA, change the embedding model device from:

```python
model_kwargs={"device": "cuda"}
```

to:

```python
model_kwargs={"device": "cpu"}
```

CPU mode may be slower, especially when using `hkunlp/instructor-xl`.

For better CPU performance, consider using a smaller embedding model such as:

```python
sentence-transformers/all-MiniLM-L6-v2
```

## Important Notes

* The app only extracts selectable PDF text.
* Scanned PDFs or image-based PDFs will not work unless OCR is added.
* FAISS indexes are stored in memory only.
* Uploaded PDFs are not permanently saved.
* Chat history only exists during the current Streamlit session.
* The app currently does not show page citations or source passages.
* Long PDFs may take longer to process.
* Large embedding models may require significant RAM or VRAM.

## Troubleshooting

### `RepositoryNotFoundError: 401 Client Error`

Check that the model name is correct.

Use:

```python
hkunlp/instructor-xl
```

not:

```python
hkunlp/instructor-x1
```

The correct name ends with lowercase `l`, not the number `1`.

Also make sure your Hugging Face token is valid.

### `ModuleNotFoundError: No module named 'torchvision'`

Install torchvision:

```bash
pip install torchvision
```

If you use CUDA, install a PyTorch and torchvision version that matches your CUDA setup.

### `AttributeError: 'INSTRUCTOR' object has no attribute '_text_length'`

This is usually caused by a version mismatch between `InstructorEmbedding` and `sentence-transformers`.

Try installing compatible versions:

```bash
pip uninstall -y sentence-transformers InstructorEmbedding
pip install sentence-transformers==2.2.2 InstructorEmbedding==1.0.1
```

Then restart Streamlit.

### `Created a chunk of size ... which is longer than the specified ...`

This means the text splitter created a chunk larger than the target chunk size.

Use `RecursiveCharacterTextSplitter`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

text_chunks = text_splitter.split_text(raw_text)
```

### `TypeError: 'NoneType' object is not callable`

This means `st.session_state.conversation` is `None`.

Make sure the user has uploaded and processed documents before asking a question.

Also make sure your conversation chain function returns the chain:

```python
return conversation_chain
```

You can also add this guard:

```python
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your documents first.")
        return

    response = st.session_state.conversation({
        "question": user_question
    })
```

### `KeyError: 'HUGGINGFACEHUB_API_TOKEN'`

Make sure your `.env` file exists in the same folder as `app.py`.

It should contain:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

Then restart Streamlit.

### CUDA is unavailable

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, either:

* install the correct CUDA-enabled PyTorch version, or
* change the embedding device to CPU.

### PDF gives poor or empty answers

The PDF may be:

* scanned
* image-based
* encrypted
* badly formatted
* using complex columns or tables

Try copying text from the PDF manually. If text cannot be selected, OCR is needed before the app can process it properly.

### FAISS installation fails

Try:

```bash
pip install faiss-cpu
```

If that fails, use a supported Python version such as Python 3.10 or 3.11 in a clean virtual environment.

## Main Functions

### `get_pdf_text(pdf_docs)`

Extracts text from all uploaded PDF files.

### `get_text_chunks(text)`

Splits extracted text into smaller overlapping chunks.

### `get_vectorstore(text_chunks)`

Creates embeddings for each chunk and stores them in a FAISS vector index.

### `get_conversation_chain(vectorstore)`

Creates the conversational retrieval chain using the FAISS retriever and chat model.

### `handle_userinput(user_question)`

Sends the user question to the conversation chain and displays the response.

### `main()`

Builds the Streamlit interface and controls the application flow.

## Security Notes

* Do not upload confidential documents unless you are allowed to send their contents to the configured model provider.
* PDF text is embedded locally, but retrieved context and questions may be sent to the Hugging Face inference endpoint.
* Keep your API token private.
* Never commit `.env` files to GitHub.
* Add authentication before deploying the app publicly.

## Future Improvements

Possible improvements include:

* Add OCR support for scanned PDFs
* Show source document names and page numbers
* Display retrieved chunks used to answer each question
* Persist FAISS indexes to disk
* Allow users to select the embedding model
* Allow users to choose CPU or CUDA from the interface
* Add file upload validation
* Add better error handling
* Add support for Word documents and text files
* Add user authentication
* Add deployment configuration
* Add automated tests
* Add logging and monitoring
