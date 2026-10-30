from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------
# Embedding Model
# -----------------------------------

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------------
# Create Vector Store
# -----------------------------------

def create_vector_store(pages):
    """
    Create page-aware chunks so PDF page citations
    remain accurate.
    """

    documents = [
        Document(
            page_content=page["text"],
            metadata={"page": page["page"]}
        )
        for page in pages
        if page["text"].strip()
    ]

    if not documents:
        raise ValueError(
            "No readable text found in the uploaded PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["order"] = index

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    return vector_store, chunks


# -----------------------------------
# Retrieve Relevant Clauses
# -----------------------------------

def retrieve_relevant_clauses(
    vector_store,
    queries,
    k=4,
    cap=20
):
    """
    Retrieve relevant chunks using FAISS MMR.

    Also include the immediately preceding and
    following chunks when they belong to the same
    PDF page.

    This helps preserve clauses split across
    chunk boundaries.
    """

    if isinstance(queries, str):
        queries = [queries]

    # Retrieve the indexed documents.
    all_documents = list(
        vector_store.docstore._dict.values()
    )

    documents_by_order = {
        doc.metadata["order"]: doc
        for doc in all_documents
    }

    # Step 1: Retrieve relevant chunks.
    selected = {}

    for query in queries:
        results = (
            vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=20
            )
        )

        for doc in results:
            order = doc.metadata["order"]
            selected[order] = doc

    # Step 2: Expand each result to neighboring
    # chunks on the same PDF page.
    expanded = dict(selected)

    for order, doc in selected.items():
        current_page = doc.metadata.get("page")

        for neighbor_order in [
            order - 1,
            order + 1
        ]:
            neighbor = documents_by_order.get(
                neighbor_order
            )

            if neighbor is None:
                continue

            if (
                neighbor.metadata.get("page")
                == current_page
            ):
                expanded[neighbor_order] = neighbor

    # Step 3: Prioritize relevant chunks and
    # their surrounding context.
    selected_orders = set(selected.keys())

    prioritized_orders = sorted(
        expanded.keys(),
        key=lambda order: (
            0 if order in selected_orders else 1,
            order
        )
    )

    # Apply the context limit.
    prioritized_orders = prioritized_orders[:cap]

    # Restore original document order.
    prioritized_orders.sort()

    return [
        documents_by_order[order]
        for order in prioritized_orders
    ]