import streamlit as st
import os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.memory import ChatMemoryBuffer

# --- 1. 網頁介面與 API 金鑰配置 ---
st.set_page_config(page_title="系統動力學智慧助教", layout="wide", page_icon="👨‍🏫")
st.title("🤖系統動力學：智慧教學系統")

# 從 Streamlit Secrets 讀取 API Key (部署後設定)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ 找不到 GOOGLE_API_KEY！請在 Secrets 中設定。")
    st.stop()

# 路徑定義
PERSIST_DIR = "./storage"
DATA_DIR = "./data"

# --- 2. 核心引擎初始化 (使用 Gemini) ---
@st.cache_resource
def init_expert_system():
    # 1. 統一使用 model_name 參數
    # 2. 移除字串前的 models/ 前綴 (避免路徑重複疊加導致 404)
    Settings.llm = Gemini(model_name="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(model_name="text-embedding-004", api_key=GOOGLE_API_KEY)
    
    # 持久化邏輯
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    course_outline = "1.建模流程 2.因果環路圖 3.存量流量圖 4.延遲與Little's Law 5.模型驗證"

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            f"你是一位專家教授。1.嚴禁說『根據教材』。2.內化知識直接回答。3.僅限系統動力學範圍。\n大綱：{course_outline}\n4.限繁體中文。"
        )
    )
    return chat_engine

chat_engine = init_expert_system()

# --- 3. 對話介面 (其餘邏輯與之前相同) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("請輸入問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt + " (請以繁體中文回答)")
        st.markdown(str(response))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})