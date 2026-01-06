import uuid
import json
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

from metrics import compute_ragas, llm_as_judge, compute_mrr


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.rerun()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except:
        return []


def extract_ai_content(chunk):
    """Extract text content from various AI message formats."""
    
    # Method 1: Direct content access
    if hasattr(chunk, 'content'):
        content = chunk.content
        
        # Handle Google Gemini's structured response
        if isinstance(content, list):
            # Extract text from each block
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'text' and 'text' in block:
                        text_parts.append(block['text'])
                    elif 'text' in block:
                        text_parts.append(block['text'])
                    elif 'parts' in block:
                        # Handle nested parts
                        parts = block.get('parts', [])
                        for part in parts:
                            if isinstance(part, str):
                                text_parts.append(part)
                elif isinstance(block, str):
                    text_parts.append(block)
            return ' '.join(text_parts) if text_parts else ''
        
        # Handle dictionary content
        elif isinstance(content, dict):
            if 'text' in content:
                return content['text']
            elif 'response' in content:
                return content['response']
            elif 'content' in content:
                return content['content']
            else:
                # Try to extract any string value
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 10:
                        return value
                return str(content)
        
        # Handle string content
        elif isinstance(content, str):
            # Try to parse as JSON
            if content.strip().startswith('{'):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        return data.get('text', data.get('content', content))
                except:
                    pass
            return content
    
    # Method 2: String representation
    return str(chunk)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# Metrics store
if "metrics_log" not in st.session_state:
    st.session_state["metrics_log"] = []

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

threads = list(dict.fromkeys(st.session_state["chat_threads"]))[::-1]
selected_thread = None


# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        # Use st.spinner() for PDF indexing
        with st.spinner("Indexing PDF..."):
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            st.sidebar.success(f"PDF indexed: {uploaded_pdf.name}")

st.sidebar.subheader("Past conversations")
for idx, thread_id in enumerate(threads):
    if st.sidebar.button(str(thread_id)[:8] + "...", key=f"thread-{thread_id}-{idx}", use_container_width=True):
        selected_thread = thread_id


# ============================ Chat UI ============================
st.title("Multi Utility Chatbot")

# Display chat history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    retrieved_context = []
    answer_parts = []

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_key}},
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                content = chunk.content
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {}

                if isinstance(content, dict) and "context" in content:
                    context_list = content["context"]
                    if isinstance(context_list, list):
                        retrieved_context.extend(context_list)
                    else:
                        retrieved_context.append(str(context_list))

            if isinstance(chunk, AIMessage):
                # Extract content using our helper function
                content = extract_ai_content(chunk)
                
                if content:
                    full_response += str(content)
                    placeholder.markdown(full_response + "▌")
                    answer_parts.append(str(content))
        
        placeholder.markdown(full_response)
        answer = full_response

    st.session_state["message_history"].append(
        {"role": "assistant", "content": answer}
    )

    # ---------------- METRICS LOGGING ----------------
    judge_score = llm_as_judge(user_input, answer, retrieved_context)
    
    # Calculate MRR for this query
    mrr_score = 0.0
    if retrieved_context:
        mrr_score = compute_mrr(retrieved_context, user_input)

    st.session_state["metrics_log"].append({
        "thread_id": thread_key,
        "query": user_input,
        "answer": answer or "",
        "judge_score": judge_score,
        "mrr_score": mrr_score,
        # stringified for UI
        "context_preview": " | ".join([str(c) for c in retrieved_context[:2]]) if retrieved_context else "",
        # raw for RAGAS - ensure it's always a list
        "raw_context": retrieved_context if isinstance(retrieved_context, list) else [],
    })


# ============================ Thread Reload ======================
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    st.session_state["message_history"] = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages
    ]
    st.rerun()


# ============================ Metrics Dashboard ==================
st.divider()
st.subheader("📊 RAG Evaluation Dashboard")

if st.session_state["metrics_log"]:
    # ---- RAGAS SAFE INPUTS ----
    questions = []
    answers = []
    contexts = []
    mrr_scores = []

    for m in st.session_state["metrics_log"]:
        raw_context = m.get("raw_context", [])
        if raw_context or m.get("answer"):
            questions.append(m["query"])
            answers.append(m["answer"])
            
            # Ensure context is properly formatted for RAGAS
            if isinstance(raw_context, list):
                clean_context = [str(item) for item in raw_context if item is not None]
                contexts.append(clean_context)
            else:
                contexts.append([str(raw_context)])
            
            # Get MRR score
            mrr_scores.append(m.get("mrr_score", 0.0))

    if questions:
        try:
            ragas_scores = compute_ragas(questions, answers, contexts)

            # Display all metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Faithfulness", round(ragas_scores["faithfulness"], 3))
            col2.metric("Answer Relevancy", round(ragas_scores["answer_relevancy"], 3))
            col3.metric("Context Precision", round(ragas_scores["context_precision"], 3))
            
            # Calculate average MRR
            avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
            col4.metric("MRR (Avg)", round(avg_mrr, 3))
            
            # Calculate average judge score
            judge_scores = [m.get("judge_score", 0) for m in st.session_state["metrics_log"] if m.get("judge_score") is not None]
            avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0
            col5.metric("LLM Judge (Avg)", round(avg_judge, 2))
            
        except Exception as e:
            st.warning(f"RAGAS evaluation error: {e}")
            
            # Still show other metrics if RAGAS fails
            col1, col2, col3 = st.columns(3)
            
            # Calculate average MRR
            avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
            col1.metric("MRR (Avg)", round(avg_mrr, 3))
            
            # Calculate average judge score
            judge_scores = [m.get("judge_score", 0) for m in st.session_state["metrics_log"] if m.get("judge_score") is not None]
            avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0
            col2.metric("LLM Judge (Avg)", round(avg_judge, 2))
            
            # Show placeholder for RAGAS
            col3.metric("RAGAS", "Config Needed")

    # ---- UI-SAFE TABLE (NO LIST COLUMNS) ----
    ui_rows = []
    for i, m in enumerate(st.session_state["metrics_log"]):
        row = {
            "thread_id": str(m["thread_id"])[:8] + "...",
            "query": m["query"][:50] + "..." if len(m["query"]) > 50 else m["query"],
            "judge_score": round(m.get("judge_score", 0), 2),
            "mrr": round(m.get("mrr_score", 0), 3),
            "context_preview": m.get("context_preview", "")[:100] + "..." if len(m.get("context_preview", "")) > 100 else m.get("context_preview", ""),
        }
        ui_rows.append(row)

    st.dataframe(ui_rows, use_container_width=True)
    
    # Detailed analysis expander
    with st.expander("📈 Detailed Analysis"):
        tab1, tab2, tab3 = st.tabs(["MRR Analysis", "Judge Scores", "Context Details"])
        
        with tab1:
            st.write("**MRR Scores per Query:**")
            for i, (query, mrr_score) in enumerate(zip(questions, mrr_scores)):
                st.write(f"**Query {i+1}:** '{query[:60]}...' → **MRR:** {mrr_score:.3f}")
            if mrr_scores:
                st.metric("Overall MRR", f"{sum(mrr_scores)/len(mrr_scores):.3f}")
        
        with tab2:
            st.write("**LLM Judge Scores per Query:**")
            for i, m in enumerate(st.session_state["metrics_log"]):
                score = m.get("judge_score", 0)
                query = m["query"]
                st.write(f"**Query {i+1}:** '{query[:60]}...' → **Score:** {score}/5.0")
        
        with tab3:
            st.write("**Retrieved Context Preview:**")
            for i, m in enumerate(st.session_state["metrics_log"]):
                context_preview = m.get("context_preview", "")
                st.write(f"**Query {i+1}:** '{m['query'][:40]}...'")
                st.write(f"**Context:** {context_preview[:150]}...")
                st.divider()

    # Clear metrics button
    if st.button("Clear Metrics Log"):
        st.session_state["metrics_log"] = []
        st.rerun()
        
else:
    st.info("No evaluation data collected yet. Ask some questions to see metrics!")