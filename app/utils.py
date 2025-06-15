from pydantic import BaseModel
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core import (
    VectorStoreIndex,
    Settings
)
from dataclasses import dataclass
from typing import List
import fitz 
import re 
import os

key = os.environ['OPENAI_API_KEY']

@dataclass
class Input:
    query: str
    file_path: str

@dataclass
class Citation:
    source: str
    text: str

class Output(BaseModel):
    query: str
    response: str
    citations: list[Citation]

class ParsedClause(BaseModel): 
    number: str 
    title: str 
    text: str 
    hierarchy: List[str]

    def to_document(self) -> Document: 
        return Document(
            text = self.text, 
            metadata = {
                "number": self.number, 
                "title": self.title, 
                "hierarchy": self.hierarchy, 
                "source": "Laws of the Seven Kingdoms" # maybe shouldn't hardcode? (TODO)
            }
        )

class DocumentService:

    """
    Update this service to load the pdf and extract its contents.
    The example code below will help with the data structured required
    when using the QdrantService.load() method below. Note: for this
    exercise, ignore the subtle difference between llama-index's 
    Document and Node classes (i.e, treat them as interchangeable).

    # example code
    def create_documents() -> list[Document]:

        docs = [
            Document(
                metadata={"Section": "Law 1"},
                text="Theft is punishable by hanging",
            ),
            Document(
                metadata={"Section": "Law 2"},
                text="Tax evasion is punishable by banishment.",
            ),
        ]

        return docs

     """
    def extract_pdf_text(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text

    def create_documents(self, file_path: str) -> List[Document]: 
        # need to convert PDF -> text 
        raw_text = self.extract_pdf_text(file_path) 
        print(raw_text) 
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # variables for keeping track of state
        clauses = []
        current_title = None 
        current_clause = None

        # match pattern for number-only lines 
        # (pdf parser splits the numbers onto their own lines)
        number_pattern = re.compile(r"^(\d+(\.\d+)*\.)$")
        n_lines = len(lines)

        idx = 0 
        while idx < n_lines: 
            line = lines[idx]
            match = number_pattern.match(line) 

            if match: 
                number = match.group(1).rstrip('.')
                parts = number.split('.')

                # just one part = top-level section = next line is title of section
                if len(parts) == 1: 
                    if idx + 1 < n_lines: 
                        current_title = lines[idx+1].strip() 
                    idx += 2 # skip title number and title 
                # multi-part number = new clause 
                else: 
                    # close previous clause if there was one 
                    if current_clause:  
                        clauses.append(current_clause)

                    # start new clause  
                    if idx + 1 < n_lines: 
                        text = lines[idx + 1].strip() 
                    else: 
                        text = ""
                    current_clause = ParsedClause(
                        number=number,
                        title=current_title or "Unknown",
                        text=text,
                        hierarchy=parts
                    )
                    idx += 2 # skip clause number and text 
            else: # in case multi-paragraph clauses ever arise 
                if current_clause: 
                    current_clause.text += " " + line 
                idx += 1 # normal step forward 

        # last one 
        if current_clause:
            clauses.append(current_clause)

        return [c.to_document() for c in clauses]

class QdrantService:
    def __init__(self, k: int = 2):
        self.index = None
        self.k = k
    
    def connect(self) -> None:
        client = qdrant_client.QdrantClient(location=":memory:")
                
        vstore = QdrantVectorStore(client=client, collection_name='temp')

        Settings.embed_model = OpenAIEmbedding(batch_size=10)
        Settings.llm = OpenAI(api_key=key, model="gpt-3.5-turbo") # TODO: switch back to "gpt-4"

        self.index = VectorStoreIndex.from_vector_store(vector_store=vstore)

    def load(self, docs = list[Document]):
        self.index.insert_nodes(docs)
    
    def query(self, query_str: str) -> Output:
        # initialize query engine & run query 
        query_engine = self.index.as_query_engine(similarity_top_k=self.k)
        response = query_engine.query(query_str)

        # create citations from response's source nodes  
        citations = []
        for node in response.source_nodes:
            number = node.metadata.get("number", "Unknown")
            title = node.metadata.get("title", "Unknown")
            citations.append(
                Citation(
                    source=f"{title}, Law {number}",
                    text=node.text.strip()
                )
            )
        # create output from response & citations 
        output = Output(
            query=query_str,
            response=response.response.strip(),
            citations=citations
        )

        return output
       

if __name__ == "__main__":
    doc_service = DocumentService()
    docs = doc_service.create_documents("docs/laws.pdf")  # <-- adjust path if needed

    print(f"Loaded {len(docs)} documents")

    qservice = QdrantService(k=3)
    qservice.connect()
    qservice.load(docs)

    query_text = "What happens if someone steals something?"
    output = qservice.query(query_text)

    print("\n--- TEST OUTPUT ---")
    print(f"Query: {output.query}")
    print(f"Response: {output.response}")
    print("Citations:")
    for c in output.citations:
        print(f" - {c.source}: {c.text}")





