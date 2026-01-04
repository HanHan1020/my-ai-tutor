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

# --- 1. 網頁 UI 與 API 配置 ---
st.set_page_config(page_title="系統動力學教授", layout="wide", page_icon="👨‍🏫")
st.title("🤖 系統動力學智慧導師系統")

# 從 Streamlit Secrets 讀取 API Key
try:
    # 確保 Secrets 中的名稱與此處一致
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("❌ 找不到 GOOGLE_API_KEY！請在 Streamlit Cloud 的 Advanced settings > Secrets 中設定。")
    st.stop()

# 定義資料路徑
PERSIST_DIR = "./storage"
DATA_DIR = "./data"

# --- 2. 核心引擎初始化 (專家角色與 Gemini API) ---
@st.cache_resource
def init_expert_system():
    # A. 模型配置 (修正後的最新相容格式)
    # 使用 model_name 避免 404 錯誤，並加入 latest 確保資源存取
    Settings.llm = Gemini(
        model_name="models/gemini-1.5-flash-latest", 
        api_key=GOOGLE_API_KEY
    )
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004", 
        api_key=GOOGLE_API_KEY
    )
    
    # B. 索引持久化邏輯
    if not os.path.exists(PERSIST_DIR):
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            st.error(f"❌ 找不到教材！請確保 GitHub 的 '{DATA_DIR}' 資料夾內有 PDF。")
            st.stop()
        with st.spinner("教授正在閱讀教材並建立知識體系..."):
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    # C. 配置對話引擎 (專家 Persona + 嚴格語言約束)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "你是一位具備 30 年教學經驗的『系統動力學權威教授』。請遵守以下最高指導原則：\n"
            "1. 語言鎖定：你只使用『台灣繁體中文』回答。嚴禁使用簡體字、大陸用語（如質量、優化、打印）。\n"
            "2. 消除冗餘：直接回答問題，嚴禁使用『根據提供的教材』、『根據上下文』等生硬開場白。將知識視為你腦中的內在智慧。\n"
            "3. 專業守則：回答僅限於系統動力學。若問題無關，請列出教學大綱（Ch 3, 5, 6, 11）並引導回課程。\n"
            "4. 結構化回答：具備學術深度，重要專有名詞加註英文。"
        )
    )
    return chat_engine

# 啟動系統
chat_engine = init_expert_system()

# --- 3. 對話介面與歷史紀錄 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 教授引用的文獻出處"):
                st.write(msg["sources"])

# 使用者輸入
if prompt := st.chat_input("向教授請教關於教材的內容..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("教授正在構思專業的回覆..."):
            # 在 prompt 後端自動追加隱形指令，強化語言鎖定
            response = chat_engine.chat(prompt + " (注意：請務必以繁體中文回答，嚴禁簡體)")
            answer = str(response)
            
            # 提取來源資訊 (用於學術驗證)
            ref_info = ""
            if hasattr(response, 'source_nodes'):
                for i, node in enumerate(response.source_nodes):
                    fname = node.metadata.get('file_name', '未知章節')
                    score = f"{node.score:.2f}" if node.score else "N/A"
                    ref_info += f"**[文獻片段 {i+1}]** `{fname}` (關聯權重: {score})\n\n"
            
            st.markdown(answer)
            if ref_info:
                with st.expander("🔍 教授引用的文獻出處"):
                    st.write(ref_info)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": ref_info
            })