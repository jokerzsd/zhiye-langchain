"""Lazy-load providers so retrieval demo never needs API credentials."""


def create_model(provider: str, model_name: str, base_url: str, api_key: str = ""):
    if not model_name.strip():
        raise ValueError("请填写模型名称。")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("服务地址须以 http:// 或 https:// 开头。")
    if provider == "online":
        if not api_key.strip():
            raise ValueError("请填写在线模型 API Key。")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name.strip(), api_key=api_key,
                          base_url=base_url, temperature=0, timeout=60, max_retries=1)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name.strip(), base_url=base_url,
                          temperature=0, client_kwargs={"timeout": 120})
    raise ValueError("未知模型提供方。")
