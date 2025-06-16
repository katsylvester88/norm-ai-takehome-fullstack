# Norm AI Takehome - Katherine Sylvester

## About 

This codebase implements an API that answers user questions about laws from the 
*Game of Thrones* universe. 

- The laws are provided in `docs/laws.pdf`, sourced from a wiki GOT fandom site
- `app/utils.py` implements the core logic to parse the PDF, chunk it into searchable Documents, embed them into a vector database (Qdrant), and use RAG to find and 
cite the most relevant clauses
- GPT-4 generates a comprehensive answer based on the retrieved context
- `app/main.py` exposes this functionality as a FastAPI service with an interactive Swagger UI.

**Note:** The `frontend/` folder is intentionally undeveloped (it wasn't a target of my takehome exercise).

## Usage Instructions
Upon cloning the repository: 

1. **Build the Docker image**

`docker build -t <container name> .`

2. **Run the container**

Ensure OPENAI_API_KEY is set as an environment variable:

`export OPENAI_API_KEY=<your api key>`

Then run the container with:

`docker run -e OPENAI_API_KEY -p 8000:80 <container name>`

3. **Query the API**
* Open http://localhost:8000/docs
* Use the /query endpoint to submit a question such as "what happens if I steal 
from the Sept?"

## Implementation Notes 

#### utils.py 
* I converted Citation to a Pydantic model so that FastAPI could serialize it 
properly as part of the Output schema 
* I used PyMuPDF / fitz for extracting the PDF as text; I found it was best for 
processing the text in order and in the correct chunks (i.e. one sentence per line instead of one word per line) unlike other packages (pdfminer, PyPDF2)
* I separated each clause into its own Document. I think this is more scalable 
(better support for longer sub-clauses), more precise, and more explainable than 
making each overall law category its own Document 
* I processed clauses in a way that handles multi-paragraph clauses, just 
because I thought that was more generalizable/realistic, although it wasn't 
necessary for this particular input. 

#### main.py 
* I set k=3 because I felt it met a good balance of detail for potentially 
complex user questions, while not providing superfluous information. In 
production this could be tuned for particular use cases

#### requirements.txt
* I downgraded the pydantic version to 1.10.x for compatibility with qdrant-client


## Design Assumptions / Other Considerations 
* Assumes that the input legal text will follow a consistent clause numbering 
convention 
* Embeddings and Qdrant are initialized in-memory for simplicity; in production 
we would need to persist / scale this 
* For more complex use cases (where more precision is required), we could add a cross-encoder for re-ranking and/or require ChatGPT to adhere strictly to the retrieved citation text. Right now, if you ask a question without an answer in the laws, 
GPT will give an answer based on its best guess 
