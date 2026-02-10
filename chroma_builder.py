# =========================================
# quick note 
# every tag is a summary of the segment where the words are separated my '-'
# the data comes from scientific articles, where text structure is important and expert-level vocabulary is used
# =========================================
import re
import chromadb
import ollama
from tqdm import tqdm

TEXT_FILE = "run.txt"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "run"
EMBED_MODEL = "nomic-embed-text"

def parse_tagged_sections(text: str):
    """
    Parses <tag>...</tag> blocks where tag may contain hyphens.
    Returns a list of dicts: {tag, content}
    """
    pattern = re.compile(
        r"<([A-Za-z0-9_-]+)>\s*(.*?)\s*</\1>",
        re.DOTALL
    )

    sections = []
    for match in pattern.finditer(text):
        sections.append({
            "tag": match.group(1),
            "content": match.group(2).strip()
        })

    return sections


def normalize_text(text: str) -> str:
    """
    Minimal, lossless normalization for scientific text.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


with open(TEXT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

sections = parse_tagged_sections(raw_text)

if not sections:
    raise ValueError("No tagged sections found in run.txt")

print(f"Found {len(sections)} tagged sections")

client = chromadb.PersistentClient(path=CHROMA_PATH)

# client.delete_collection(COLLECTION_NAME) # for dv rebuilding

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

for idx, s in enumerate(tqdm(sections, desc="Indexing sections")):
    tag = s["tag"]
    content = normalize_text(s["content"])

    embedding_input = f"Section: {tag}\n\n{content}"

    embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=embedding_input
    )["embedding"]

    collection.add(
        ids=[f"{tag}_{idx}"],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{"tag": tag,"source": TEXT_FILE}])

print("Ingestion complete.")
print("Total vectors stored:", collection.count())