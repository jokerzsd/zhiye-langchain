from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_load_ask_export_and_clear():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label.startswith("载入示例")).click().run()
    assert len(app.metric) == 3
    app.chat_input[0].set_value("图书馆几点关门？").run()
    assert not app.exception
    assert "22:00" in app.session_state["history"][0]["answer"]
    assert app.session_state["history"][0]["sources"]
    next(b for b in app.button if b.label == "清空资料与问答").click().run()
    assert not app.exception
    assert len(app.metric) == 0
