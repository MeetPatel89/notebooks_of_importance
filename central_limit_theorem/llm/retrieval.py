import faiss
import torch
from transformers import AutoModel, AutoTokenizer


class RetrievalModule:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Initialize the FAISS index (flat index)
        self.dimension = 384  # all-MiniLM-L6-v2 output vector dimension
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store raw documents for reference
        self.documents = []

    def encode(self, texts):
        """Encode texts into vectors using the transformer model."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def add_documents(self, docs):
        """Add documents to the FAISS index."""
        vectors = self.encode(docs)
        self.index.add(vectors)
        self.documents.extend(docs)  # Save the mapping of raw documents

    def retrieve(self, query, top_k=5):
        """Retrieve top-k documents matching the query."""
        query_vector = self.encode([query])
        distances, indices = self.index.search(query_vector, top_k)
        return [(self.documents[idx], distances[0][i]) for i, idx in enumerate(indices[0])]


# 📋 Usage
if __name__ == "__main__":
    # Initialize retrieval module
    retrieval_module = RetrievalModule()

    # Add some example documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "AI is transforming the tech industry.",
        "Machine learning is a subset of AI.",
        "Quantum computing is the future of computing.",
        "Understanding the universe is humanity's ultimate quest.",
    ]
    retrieval_module.add_documents(documents)

    # Retrieve similar documents
    query = "The future of AI"
    top_documents = retrieval_module.retrieve(query, top_k=3)

    print("Query:", query)
    print("Top similar documents:")
    for doc, distance in top_documents:
        print(f"Document: {doc} (Distance: {distance:.4f})")
