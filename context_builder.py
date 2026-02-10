import chromadb
import ollama

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME_RUN = "run"
COLLECTION_NAME_RUNNER = "runner"
EMBED_MODEL = "nomic-embed-text"

def retrieve_relevant_sections(query: str,  collection_name = COLLECTION_NAME_RUN, k: int = 3, preffix: str = ""):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=collection_name) 
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    retrieved = []
    retrieved.append({"content": preffix,"tag": preffix,"distance": 0})
    
    for i in range(len(results["documents"][0])):
        retrieved.append({"content": results["documents"][0][i],"tag": results["metadatas"][0][i]["tag"],"distance": results["distances"][0][i] })

    return retrieved


def build_rag_context(sections):
    """
    Turns retrieved sections into a structured context block.
    """
    blocks = []
    for s in sections:
        blocks.append( f"[SECTION: {s['tag']}]\n{s['content']}" )
    return "\n\n---\n\n".join(blocks)


def build_prompt(user_query: str, context: str):
    return f"""
You are an expert assistant, specializing in running mechanics.
the section CONTEXT contains detailed information about the runner 
the section CONTEXT contains detailed information running mechanics 
running mechanics
<rules>
(1) Answer the user's question using ONLY the information provided below.
(2) Choose the most relevant information.
(3) Provide an easy to understand answer.
(4) Keep the answer short and structured.
(5) If the information is insufficient, say so explicitly.
</rules>
Follow the 'rules' (1), (2), (3), (4), (5)

CONTEXT:
{context}

USER QUESTION:
{user_query}

""".strip()

if __name__ == "__main__":
    query = "How does fatigue affect running cadence late in a race?"

    sections_run = retrieve_relevant_sections(query,COLLECTION_NAME_RUN)
    sections_runner = retrieve_relevant_sections(query,COLLECTION_NAME_RUNNER)
    sections = sections_run + sections_runner
    context = build_rag_context(sections)
    prompt = build_prompt(query, context)

    print(prompt)