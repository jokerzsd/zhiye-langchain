"""Run with: streamlit run app.py"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from zhiye.core import KnowledgeBase, answer_question, load_document
from zhiye.models import create_model

load_dotenv()
st.set_page_config(page_title="知页 · 智能资料问答", page_icon="📚", layout="wide")
st.markdown("<style>.block-container{max-width:1120px;padding-top:2.5rem}"
            "h1{letter-spacing:-1px}div[data-testid=stMetric]{background:#EDF2F7;"
            "padding:18px;border-radius:12px}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 知页")
    st.caption("让每一个答案，都有资料可查。")
    mode = st.selectbox("运行模式", ["检索演示（无需密钥）", "在线大模型", "本地 Ollama"])
    model_name, base_url, api_key = "", "", ""
    if mode == "在线大模型":
        model_name = st.text_input("模型名称", os.getenv("OPENAI_MODEL", ""), placeholder="填写服务商提供的模型 ID")
        base_url = st.text_input("API 地址", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        api_key = st.text_input("API Key", os.getenv("OPENAI_API_KEY", ""), type="password")
        st.caption("问题和命中的资料片段将发送至所填地址。")
    elif mode == "本地 Ollama":
        model_name = st.text_input("本地模型", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
        base_url = st.text_input("Ollama 地址", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        st.caption("请先启动 Ollama，并下载同名模型。本地地址可用于离线问答。")
    top_k = st.slider("引用片段数量", 1, 8, 4)
    st.divider()
    if st.button("清空资料与问答", use_container_width=True):
        for key in ("kb", "history", "source_count"):
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("资料仅保存在当前应用会话内存中。问答按单轮处理，请在每次提问中写全主题。")

st.caption("LANGCHAIN  /  RETRIEVAL AUGMENTED GENERATION")
st.title("把资料，变成可以提问的知识库")
st.write("上传学习资料、制度手册或项目文档，快速找到答案和原文依据。")

with st.expander("① 准备资料", expanded="kb" not in st.session_state):
    uploads = st.file_uploader("支持 PDF / Markdown / TXT，最多 10 个文件，每个不超过 10 MB", type=["pdf", "md", "txt"], accept_multiple_files=True)
    left, right = st.columns(2)
    build = left.button("建立资料库", type="primary", use_container_width=True)
    sample = right.button("载入示例 · 星河学院手册", use_container_width=True)
    if build or sample:
        try:
            if sample:
                file = Path(__file__).parent / "examples" / "campus_handbook.md"
                docs = load_document(file.name, file.read_bytes())
            else:
                if not uploads or len(uploads) > 10:
                    raise ValueError("请选择 1–10 个文件。")
                docs = []
                total_chars = 0
                for upload in uploads:
                    parsed = load_document(upload.name, upload.getvalue())
                    total_chars += sum(len(doc.page_content) for doc in parsed)
                    if total_chars > 1_000_000:
                        raise ValueError("资料库提取文字合计不能超过 100 万字。")
                    docs.extend(parsed)
            st.session_state.kb = KnowledgeBase(docs)
            st.session_state.source_count = len({d.metadata["source"] for d in docs})
            st.session_state.history = []
            st.success("资料库已就绪，可以开始提问。")
        except ValueError as exc:
            st.error(str(exc))

if "kb" not in st.session_state:
    st.info("先载入示例即可体验。示例问题：图书馆几点关门？奖学金需要哪些申请材料？")
    st.stop()

kb = st.session_state.kb
columns = st.columns(3)
columns[0].metric("资料文件", st.session_state.source_count)
columns[1].metric("可检索片段", len(kb.chunks))
columns[2].metric("当前模式", "检索演示" if mode.startswith("检索") else mode)
st.subheader("② 向资料提问")
st.caption("例如：图书馆几点关门？ · 奖学金需要哪些申请材料？ · 请假 5 天由谁审批？")


def render_sources(sources):
    for i, source in enumerate(sources, 1):
        meta = source["metadata"]
        page = f" · 第 {meta['page']} 页" if "page" in meta else ""
        with st.expander(f"[{i}] {meta['source']}{page} · 片段 {meta['chunk_id']}"):
            st.text(source["text"])
            st.caption(f"检索匹配分数：{source['score']:.3f}（非可信度概率）")


for item in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        render_sources(item["sources"])

question = st.chat_input("输入关于资料的问题（最多 2000 字）", max_chars=2000)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索与整理依据…"):
                model = None
                if mode != "检索演示（无需密钥）":
                    model = create_model("online" if mode == "在线大模型" else "ollama", model_name, base_url, api_key)
                answer, hits = answer_question(kb, question, model, top_k)
            sources = [{"text": h.document.page_content, "metadata": h.document.metadata, "score": h.score} for h in hits]
            st.markdown(answer)
            render_sources(sources)
            st.session_state.setdefault("history", []).append({"question": question, "answer": answer, "sources": sources})
            st.session_state.history = st.session_state.history[-20:]
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            # Avoid echoing provider exceptions, which may contain credentials or document content.
            st.error("模型调用失败。请检查模型名称、密钥、服务地址及网络；本地模式请确认 Ollama 已启动。")

history = st.session_state.get("history", [])
if history:
    exported = []
    for item in history:
        exported.append(f"## 问题\n{item['question']}\n\n## 回答\n{item['answer']}\n\n## 参考资料")
        for i, source in enumerate(item["sources"], 1):
            exported.append(f"[{i}] {source['metadata']['source']}\n\n{source['text']}")
    st.download_button("导出问答记录（Markdown）", "\n\n".join(exported), file_name="知页问答记录.md", mime="text/markdown")
