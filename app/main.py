from fastapi import FastAPI, Query
from app.utils import DocumentService, QdrantService, Output

app = FastAPI()

doc_service = DocumentService() 
qdrant_service = QdrantService(k=3)

@app.on_event("startup")
def startup_event():
    print("loading PDF and setting up Qdrant")
    docs = doc_service.create_documents("docs/laws.pdf")
    qdrant_service.connect()
    qdrant_service.load(docs)
    print(f"loaded {len(docs)} documents into Qdrant")

@app.get("/query", response_model=Output)
def run_query(query: str = Query(..., description="Your GOT legal question")):
    output = qdrant_service.query(query)
    return output