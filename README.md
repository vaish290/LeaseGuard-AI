# LeaseGuard AI 🏠

### Understand your lease before you sign, renew, or move out.

Lease agreements can be long, confusing, and full of legal language that's difficult to understand. Important details about early termination, security deposits, late fees, or additional charges are often spread across multiple sections of the document.

I built **LeaseGuard AI** to make it easier for renters to understand what their lease actually says.

Instead of manually going through pages of legal text, users can upload their lease agreement, ask questions in plain English, and receive explanations supported by references to the original document.

The goal isn't to replace legal advice. It's to help renters find relevant information, understand their responsibilities, and know what questions to ask before making a decision.

---

## What Can LeaseGuard AI Do?

- **Upload a lease:** Upload a PDF lease agreement for analysis.
- **Ask questions naturally:** Ask about subletting, security deposits, late payments, pets, property damage, and early termination.
- **Find relevant clauses:** Search the agreement for sections related to the question, including conditions, exceptions, and financial obligations.
- **Explain complex terms:** Turn formal lease language into simpler, easier-to-understand explanations.
- **Show source references:** Include PDF page references so users can verify the information themselves.
- **Highlight uncertainties:** Point out information that isn't clear from the retrieved passages and suggest questions to clarify with the landlord.

## How It Works

LeaseGuard AI uses **Retrieval-Augmented Generation (RAG)** to answer questions based on the uploaded document rather than relying only on an AI model's general knowledge.

**1. Upload the document**

The user uploads a PDF through the Streamlit interface. PyMuPDF extracts readable text while preserving PDF page information.

**2. Process the lease**

The extracted text is divided into smaller, overlapping chunks. Each chunk retains page metadata to help connect answers to the document.

**3. Create embeddings**

The chunks are converted into vector embeddings using `all-MiniLM-L6-v2` and stored in a FAISS vector index.

**4. Retrieve relevant information**

The application uses semantic search with Maximal Marginal Relevance (MMR) to retrieve relevant passages. Nearby chunks are also considered to preserve context.

**5. Generate an explanation**

The retrieved passages and the user's question are sent to an LLM through the Groq API. The model is instructed to explain relevant obligations and conditions using only the retrieved evidence.

**6. Display the answer**

The application presents an explanation, PDF page references, unresolved uncertainties, and practical questions to clarify.

### Architecture

```text
         DOCUMENT PROCESSING
            PDF Upload
                ↓
            PyMuPDF Text Extraction
                ↓
            Page-Aware Text Chunking
                ↓
            Sentence Transformers (Embeddings)
                ↓
            FAISS Vector Index
            
            
            QUESTION ANSWERING
            User Question
                ↓
            FAISS Similarity Search (MMR)
                ↓
            Relevant Chunks + Neighboring Context
                ↓
            Prompt + Retrieved Passages
                ↓
            Groq LLM (GPT-OSS-20B)
                ↓
            Source ID → PDF Page Mapping
                ↓
            Answer Displayed in Streamlit
```

---

## Tech Stack

| Technology                         | Purpose                      |
| ---------------------------------- | ---------------------------- |
| Python                             | Application development      |
| Streamlit                          | Web interface                |
| PyMuPDF                            | PDF text extraction          |
| LangChain                          | RAG pipeline integration     |
| FAISS                              | Vector storage and retrieval |
| Hugging Face Sentence Transformers | Text embeddings              |
| Groq API                           | LLM inference                |
| `openai/gpt-oss-20b`               | Answer generation            |

## Project Structure

```text
LeaseGuard-AI/
│
├── sample_data/
│   ├── LeaseGuard_Anonymized_Searchable.pdf
│   └── Unit3_Anonymized_Searchable.pdf
│
├── app.py                 # Streamlit user interface
├── lease_analyzer.py      # LLM integration, prompts, and citation handling
├── lease_schemas.py       # Pydantic models for structured lease analysis
├── pdf_processor.py       # PDF text extraction
├── rag_pipeline.py        # Text chunking, embeddings, FAISS, and retrieval
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git
└── README.md              # Project documentation

Sample Data: The sample_data/ folder contains anonymized, searchable lease PDFs that reviewers can use to test the application without needing their own lease documents.
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vaish290/LeaseGuard-AI.git
cd LeaseGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

You can obtain an API key from [Groq Console](https://console.groq.com/).

Never commit your actual API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL, upload a searchable lease PDF, and ask a question.

---

## Example Questions and Testing

I tested LeaseGuard AI using an anonymized lease agreement and several common renter questions.

The application produced useful responses during initial testing for questions such as:

| Question                                            | Information explored                                     |
| --------------------------------------------------- | -------------------------------------------------------- |
| Can I sublet my apartment?                          | Approval requirements, fees, and tenant responsibilities |
| When will I get my security deposit back?           | Return deadlines and potential deductions                |
| What happens if I pay rent late?                    | Late fees and payment consequences                       |
| Are pets allowed, and are there additional charges? | Pet deposits, fees, and restrictions                     |
| What happens if I damage the apartment?             | Damage-related responsibilities and potential charges    |

Other supported questions include early termination, moving-out responsibilities, and lease-related financial obligations. Answers depend on the uploaded agreement, and complex questions involving multiple clauses may require additional verification.

---

## Challenges and What I Learned

One of the biggest challenges was making sure the AI didn't miss important details appearing in different parts of a lease.

For example, a lease might describe an obligation in one section but include additional financial conditions in a separate addendum. Retrieving only the primary clause can result in an incomplete answer.

To improve this, I experimented with:

- **Overlapping text chunks** to preserve context.
- **MMR retrieval** to reduce repetitive results.
- **Neighboring-chunk retrieval** to capture supporting information.
- **Prompt engineering** to emphasize conditions, exceptions, and financial consequences.
- **Citation mapping** to connect answers to PDF pages.

Another challenge was balancing retrieval coverage with API token limits. Including more context can improve completeness, but it also increases inference costs and the possibility of rate-limit errors.

This project helped me understand that building a reliable RAG application involves more than connecting a vector database to an LLM. Retrieval quality, document structure, context selection, and source verification are equally important.

---

## Current Limitations

LeaseGuard AI is a prototype, and some limitations remain:

- **Retrieval completeness:** Related clauses in distant sections or addenda may not always be retrieved.
- **PDF compatibility:** Searchable PDFs are supported; image-only PDFs require OCR, which is not currently implemented.
- **Citation verification:** Source IDs are mapped to PDF pages, but individual claims are not independently checked against the cited text.
- **API limits:** Groq rate limits can affect repeated testing and larger requests.
- **Legal interpretation:** The application cannot determine whether a lease provision is legally enforceable.

## Future Improvements

- Hybrid keyword and semantic search.
- More reliable cross-section retrieval.
- OCR support for scanned documents.
- Automated evaluation using expected answers.
- Stronger citation verification.
- Better token optimization and rate-limit handling.

---

## Privacy

Lease agreements may contain personal and financial information.

The application processes uploaded documents for retrieval, and relevant extracted passages are sent to the configured Groq API for answer generation.

Users should avoid uploading sensitive documents without understanding how their information is processed. Anonymized or synthetic leases are recommended for demonstrations.

---

## AI Assistance

I used AI tools during development for guidance, debugging, exploring different implementation approaches, and helping with project documentation. AI was particularly useful for understanding alternative ways to structure the RAG pipeline and troubleshoot issues during testing.

I reviewed and adapted the suggestions while working on the implementation, integration, and testing of the application.

---

## Disclaimer

LeaseGuard AI is an educational prototype and does not provide legal advice. Users should verify important information against their original agreement and consult a qualified professional when necessary.

---

**Built using Python, Streamlit, LangChain, FAISS, and Groq, powered by Retrieval-Augmented Generation (RAG).**
