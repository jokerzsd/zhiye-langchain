import io
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfWriter

from zhiye.core import KnowledgeBase, NO_EVIDENCE, answer_question, load_document, split_documents

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kb():
    return KnowledgeBase(load_document("handbook.md", (ROOT / "examples/campus_handbook.md").read_bytes()))


def test_chinese_retrieval_and_source(kb):
    hits = kb.search("图书馆几点关门")
    assert hits and "22:00" in hits[0].document.page_content
    assert hits[0].document.metadata["source"] == "handbook.md"


def test_no_evidence_does_not_call_model(kb):
    def forbidden(_):
        pytest.fail("Model should not be called without evidence")
    answer, hits = answer_question(kb, "quantumxyz", forbidden)
    assert answer == NO_EVIDENCE and hits == []


def test_real_langchain_chain_with_fake_model(kb):
    model = FakeListChatModel(responses=["图书馆通常在 22:00 关门。[1]"])
    answer, hits = answer_question(kb, "图书馆几点关门", model)
    assert "22:00" in answer and "[1]" in answer and hits


def test_demo_explicitly_labeled(kb):
    answer, hits = answer_question(kb, "奖学金")
    assert "未调用大模型" in answer and hits


@pytest.mark.parametrize("name,data", [("a.txt", b""), ("a.exe", b"x"), ("a.txt", b"\xff"), ("a.pdf", b"broken"), ("a.md", b"  ")])
def test_invalid_upload(name, data):
    with pytest.raises(ValueError):
        load_document(name, data)


def test_pdf_without_text():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="OCR"):
        load_document("scan.pdf", buffer.getvalue())


def test_chunks_keep_metadata_and_boundaries():
    docs = load_document("long.txt", ("测试文本。" * 600).encode())
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert all(len(d.page_content) <= 700 for d in chunks)
    assert all(d.metadata["source"] == "long.txt" for d in chunks)
    assert all("start_index" in d.metadata for d in chunks)


@pytest.mark.parametrize("question", [" ", "长" * 2001])
def test_invalid_question(kb, question):
    with pytest.raises(ValueError):
        answer_question(kb, question)
