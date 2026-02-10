import streamlit as st
import pandas as pd
import ollama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from context_builder import ( retrieve_relevant_sections, build_rag_context, build_prompt)

LLM_MODEL = "llama3.2"
FOOTPATH_CSV = "footpath_representation.csv"

@st.cache_data
def load_footpath():
    return pd.read_csv(FOOTPATH_CSV)
footpath_df = load_footpath()

llm = ChatOllama( model=LLM_MODEL, temperature=0.2)

def agent_pipeline(user_query: str):
    # Data Retrieval from ChromaDB database
    sections_run =   retrieve_relevant_sections(user_query, "run", k=3, preffix="RUNNING DATA:")
    sections_runner =   retrieve_relevant_sections(user_query, "runner", k=3, preffix="RUNNER PERSONAL DATA:")
    rag_context = build_rag_context(sections_run + sections_runner)

    # 1. Thinking: analyze the data and form conclusions
    context_prompt = build_prompt(user_query, rag_context)
    print(f"Thinking...")
    thinking_prompt = ("You are an expert reasoning agent.\n"
            "Your task is to analyze retrieved data and form conclusions.\n"
            "DO NOT answer the user.\n"
            "Think step-by-step and explain what the data implies.\n") + context_prompt
        
    thinking = llm.invoke(thinking_prompt).content.strip()
    print(f"{thinking}")
    
    # 2. Evaluation: check if the thinking is logical and relevant to the question
    print(f"Evaluating...")
    evaluation_prompt = ("You are a critical evaluator.\n"
            "Check if the conclusion logically follows from the evidence.\n"
            "Identify missing data or uncertainty.\n") +   thinking
    evaluation = llm.invoke(evaluation_prompt).content.strip()
    print(f"{evaluation}")

    # 3. Generate a conclusion, based on the data, the thinking and the evaluation
    final_prompt = (
        "You are an expert assistant speaking to a runner.\n"
        "Answer clearly, concisely, and practically.\n"
        "Do NOT mention internal reasoning.\n"
        "Base your answer ONLY on validated conclusions.\n")  + context_prompt + thinking + evaluation
    
    print(f"Generating answer...")
    final_answer = llm.invoke(final_prompt).content.strip()
    print(f"{final_answer}")
    
    return {
        "context": context_prompt,
        "thinking": thinking,
        "evaluation": evaluation,
        "answer": final_answer
    }


st.set_page_config(page_title="🐯 TORA", layout="wide")
st.title("🐯 TORA")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat display
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if query := st.chat_input("Ask about running mechanics, fatigue, cadence…"):
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("📝 Analyzing biomechanics..."):
        result = agent_pipeline(query)

    # Assistant response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])


    # Thinking panel
    with st.expander("🧠 Agent Thinking & Tool Use"):
        st.subheader("Retrieved Context")
        st.text(result["context"])

        st.subheader("Reasoning")
        st.text(result["thinking"])

        st.subheader("Evaluation")
        st.text(result["evaluation"])
        
    st.session_state.chat.append({ "role": "assistant",  "content": result["answer"]})