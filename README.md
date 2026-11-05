# 知页 · 基于 LangChain 的智能资料问答应用

> 将文档变成可以提问的知识库。上传资料，检索相关内容，由在线或本地大模型整理答案，并查看原文依据。

**开源仓库：** [jokerzsd/zhiye-langchain](https://github.com/jokerzsd/zhiye-langchain)

**项目类型：** LangChain + RAG 智能应用

**开源许可证：** MIT

## 1. 项目简介

学习资料、校园制度和项目文档往往篇幅较长，人工查找答案效率较低。知页提供中文 Web 界面，让用户导入自己的资料并用自然语言提问。应用先在本地检索相关文档片段，再通过 LangChain 编排大模型生成带引用编号的回答，用户可展开引用核对原文。

应用实现三种模式：

| 模式 | 所需条件 | 输出行为 |
| --- | --- | --- |
| 检索演示 | 无需密钥、无需模型 | 展示本地检索到的原文，明确标注未调用大模型 |
| 在线大模型 | 可用的 API Key、模型名称、服务地址与网络 | 使用 `ChatOpenAI` 调用符合标准 Chat Completions 接口的服务生成答案 |
| 本地 Ollama | 已安装 Ollama，已下载模型并启动服务 | 使用 `ChatOllama` 调用本地大模型，环境准备完成后可离线运行 |

检索演示用于检查资料处理流程。要体验大模型生成回答，请选择在线模式或本地 Ollama 模式。

## 2. 功能清单

- 导入 UTF-8 编码的 TXT、Markdown，以及可提取文字的 PDF。
- 保留文件名、PDF 页码、片段编号和片段起始字符位置。
- 使用 LangChain 递归文本切分器，按中文标点与段落切分长文。
- 使用本地 BM25 词法检索，支持中文双字切分与英文单词，无需下载向量模型。
- 使用 LangChain LCEL 管道连接提示词、聊天模型和输出解析器。
- 回答附带引用编号，展开即可查看对应原文与检索分数。
- 完全没有检索命中时直接提示补充资料，不调用大模型。
- 提供虚构校园手册与示例问题，支持导出 Markdown 问答记录。
- 可清空当前资料和历史记录；最多保留最近 20 条成功问答。

## 3. 技术方案

```mermaid
flowchart LR
    A[上传 PDF / TXT / MD] --> B[内存解析与元数据]
    B --> C[LangChain 递归切分]
    C --> D[本地 BM25 索引]
    Q[用户问题] --> D
    D --> E{相关片段}
    E -->|无命中| F[提示资料不足]
    E -->|命中| G[ChatPromptTemplate]
    G --> H[ChatOpenAI / ChatOllama]
    H --> I[StrOutputParser]
    I --> J[回答与引用原文]
```

核心生成链位于 `zhiye/core.py`：

```python
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
```

| 组件 | 作用 |
| --- | --- |
| Streamlit | 上传、参数配置、聊天与记录导出 |
| langchain-core | Document、ChatPromptTemplate、LCEL、StrOutputParser |
| langchain-text-splitters | RecursiveCharacterTextSplitter |
| langchain-openai | 在线聊天模型适配 |
| langchain-ollama | 本地聊天模型适配 |
| pypdf | 提取 PDF 文字及页码 |
| 自实现 BM25 | 对文档片段进行本地词法召回 |

默认每个片段不超过 700 字符，目标重叠为 100 字符，召回 4 个片段，可在界面调为 1–8 个。BM25 使用 k1=1.5、b=0.75。检索分数只表示词法匹配程度，不是答案准确率或可信度概率。

## 4. 安装与启动

建议使用 **Python 3.11**；开发和自动化测试采用该版本。

```bash
git clone https://github.com/jokerzsd/zhiye-langchain.git
cd zhiye-langchain
python -m venv .venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装并启动（在项目根目录执行）：

```bash
python -m pip install -r requirements-lock.txt
python -m streamlit run app.py
```

浏览器访问 [http://localhost:8501](http://localhost:8501)。若 PowerShell 禁止激活脚本，可直接执行 `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt` 和 `.\.venv\Scripts\python.exe -m streamlit run app.py`。

`requirements-lock.txt` 锁定本次验收环境的依赖版本，包含测试依赖。`requirements.txt` 为运行依赖的兼容范围，更新依赖时应重新运行测试。

## 5. 三种模式使用方法

### 5.1 无需密钥的快速演示

1. 保持左侧“检索演示（无需密钥）”。
2. 点击“载入示例 · 星河学院手册”。
3. 输入“图书馆几点关门？”。
4. 查看检索原文，展开引用，确认含有 `08:00–22:00` 和考试周延长安排。
5. 输入“奖学金需要哪些申请材料？”，检查命中片段。
6. 点击“导出问答记录（Markdown）”保存记录。

示例手册是虚构数据，不代表真实学校政策。演示模式直接展示原文，不声称完成大模型推理。

### 5.2 在线大模型

将 `.env.example` 复制为 `.env`，填写：

```dotenv
OPENAI_API_KEY=填写你自己的密钥
OPENAI_MODEL=填写服务商提供的模型名称
OPENAI_BASE_URL=https://api.openai.com/v1
```

在界面选择“在线大模型”，检查模型名称、API 地址和密钥，加载资料后提问。也可直接在侧边栏填写，无需创建 `.env`。

应用通过 `ChatOpenAI` 使用标准聊天接口。第三方服务需兼容此接口；其私有字段和扩展能力不在本项目支持范围内。问题和召回片段会发送给所配置的服务，请使用有权处理的资料。实际调用费用、配额及模型可用性由服务商决定。

### 5.3 本地 Ollama 大模型

先从 [Ollama 官网](https://ollama.com/) 安装 Ollama，再联网下载模型，例如：

```bash
ollama pull qwen2.5:7b
ollama list
```

启动 Ollama 服务；桌面版通常会自动启动，未启动时可运行 `ollama serve`。在应用中选择“本地 Ollama”，模型名填写 `qwen2.5:7b`，地址填写 `http://localhost:11434`。模型名称必须与本地下载的 tag 一致。

本地模式不需要 API Key。首次安装依赖和下载模型需要网络；完成后，使用本机 Ollama 地址可在离线环境中问答。7B 模型对内存和显存有一定要求，可按设备能力选用其他 Ollama 模型。生成质量与速度取决于所选模型和硬件。

## 6. 示例问答与验收步骤

以下是**预期行为**，不是已测量的大模型性能：

| 输入 | 预期 |
| --- | --- |
| 图书馆几点关门？ | 命中图书馆时间；大模型模式应说明通常 22:00、考试周 23:00，并标注引用 |
| 奖学金需要哪些申请材料？ | 命中成绩单、个人总结、辅导员推荐意见 |
| 请假 5 天由谁审批？ | 命中超过 3 天需由学院审批的条款 |
| quantumxyz | 无关键词命中，提示资料不足，且不调用模型 |
| 上传空文件或损坏 PDF | 给出文件错误提示 |

验收建议：先完成无密钥演示，再配置一个实际模型执行相同问题，核对回答与引用是否一致；最后导出问答记录。每次问题都需包含完整主题，例如使用“图书馆考试周几点关门”，而不是依赖上下文的“那考试周呢”。

## 7. 测试与验证边界

```bash
python -m pytest -q
```

测试覆盖中文检索与来源、无命中时不调用模型、LangChain 生成链、演示标签、非法文件、无文字 PDF、切分元数据、问题长度以及 Streamlit 页面操作。生成链测试使用 LangChain 的 `FakeListChatModel`，不消耗付费额度。GitHub Actions 在 push / pull request 时自动运行测试。

本次交付在 Windows / Python 3.11 环境中运行结果为 **20 passed**；依赖一致性检查通过。测试也验证了两种模型适配器的构造和配置校验，但未向真实大模型发起推理请求。

**真实外部模型验证边界：** 交付环境未配置在线模型密钥和本地 Ollama 模型，因此不将自动化测试通过等同于真实模型推理通过。在线和本地模式需由使用者提供可用模型后完成实际推理验收。本项目未报告准确率、响应时延或规模化并发性能。

## 8. 目录结构

```text
zhiye-langchain/
├── app.py                     # Streamlit 页面
├── zhiye/
│   ├── core.py                # 解析、切分、BM25、LangChain 生成链
│   └── models.py              # 在线 / Ollama 模型工厂
├── examples/campus_handbook.md # 虚构示例资料
├── tests/                     # 核心与页面测试
├── .github/workflows/tests.yml
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── pytest.ini
├── LICENSE
└── README.md
```

## 9. 限制与数据处理

- 使用 BM25 词法召回，不是向量语义检索；同义改写、跨文档推理和无关键词问题可能漏检。后续可增加向量检索与重排序。
- 无命中会直接拒答；有词语重合不代表有充分依据，模型仍可能误答或生成错误引用，需要核对原文。提示词中要求资料不足时拒答，但不提供正确性保证。
- 通过系统提示词限制资料中的指令执行，不具备完整的提示注入防护机制。
- PDF 仅提取文本，不支持 OCR、图片、复杂表格结构或加密 PDF；PDF 最多 200 页。
- 每个文件最多 10 MB、提取文字最多 50 万字符；最多 10 个文件，整个资料库提取文字合计最多 100 万字符。
- 上传内容在当前 Streamlit 会话内存中处理，不主动写入磁盘、不建立持久化数据库；关闭或重启应用后需重新加载。下载记录由用户保存。
- 本地处理不等于所有模式都不联网：在线模式会传送问题和相关片段；Ollama 配为远程地址也会向该地址传送数据。
- `.env` 已被 Git 忽略。不要提交密钥、私人资料或真实问答记录；不启用 LangSmith 追踪。
- 默认仅监听本机；目前没有账号系统、并发隔离保证、生产级鉴权和持久化能力，适用于本机课程演示。

## 10. 常见问题

**启动提示模块不存在？** 请确认安装依赖与启动使用同一个虚拟环境，并在项目根目录启动。

**PDF 提示没有文字？** 可能是扫描件，请先在其他工具中 OCR，再导出带文本层的 PDF。

**在线模型调用失败？** 检查密钥、余额、模型名、地址和网络；地址通常应包含 `/v1`。应用不会在页面回显可能含敏感数据的底层服务异常。

**Ollama 连接失败？** 确认服务已启动，运行 `ollama list` 检查模型 tag，确认地址可访问。

**回答找不到相关内容？** 补充对应文档，使用文档原词提问，或增加召回片段数量。此版本不使用语义向量模型。

## 11. 参考资料

- [LangChain：ChatOpenAI 集成](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [LangChain：ChatOllama 集成](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [LangChain：递归文本切分](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [Streamlit 文档](https://docs.streamlit.io/)
- [Ollama](https://ollama.com/)

## 12. 课程提交说明

可将本文件作为智能应用 README 提交，并附上顶部的开源仓库链接。仓库包含完整源码、示例资料、依赖清单、自动化测试和 MIT 许可证。若课程平台要求截图或运行演示，请按第 6 节连接真实模型后补充对应材料。
