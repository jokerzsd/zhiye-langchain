import pytest

from zhiye.models import create_model


def test_online_provider_constructs_without_request():
    from langchain_openai import ChatOpenAI
    model = create_model("online", "test-model", "http://localhost:9999/v1", "not-a-real-key")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "test-model"


def test_ollama_provider_constructs_without_request():
    from langchain_ollama import ChatOllama
    model = create_model("ollama", "qwen2.5:7b", "http://localhost:11434")
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen2.5:7b"


@pytest.mark.parametrize("provider,name,url,key", [
    ("online", "", "https://example.com/v1", "key"),
    ("online", "test", "invalid", "key"),
    ("online", "test", "https://example.com/v1", ""),
    ("other", "test", "http://localhost:11434", ""),
])
def test_configuration_validation(provider, name, url, key):
    with pytest.raises(ValueError):
        create_model(provider, name, url, key)
