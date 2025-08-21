# RAG Pipeline

這是一個完整的 RAG (Retrieval-Augmented Generation) pipeline 實現，包含 retriever、reranker 和 generator 三個主要組件。

## 功能特色

- 🔍 **Retriever**: 基於向量相似度檢索相關文件
- 🔄 **Reranker**: 使用 reranker 模型重新排序檢索結果（可選）
- 🤖 **Generator**: 基於檢索到的文件生成繁體中文回答
- 📚 **向量儲存**: 使用 Chroma 持久化向量儲存
- ✂️ **文本分塊**: 智能文本分塊，支援重疊和語義完整性
- 🔌 **API 整合**: 整合現有的 embedding 和 reranker API
- ⚙️ **靈活配置**: 可選擇啟用或禁用 reranker 階段和文本分塊
- 🇹🇼 **中文支援**: 內建 system prompt 確保繁體中文回答

## 架構圖

```
用戶查詢 → Retriever → [Reranker] → Generator → 回答
              ↓           ↓           ↓
          向量搜尋    重新排序    生成回答
                    (可選)
```

## 安裝

1. 克隆專案：
```bash
git clone <repository-url>
cd rag-pipeline
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 設定環境變數（必需）：
```bash
# 方法一：使用設定腳本（推薦）
python setup_env.py

# 方法二：手動創建 .env 檔案
echo "API_KEY=your_api_key_here" > .env
```

**重要**: 
- API Key 必須設定在 `.env` 檔案中，不能通過命令行參數傳遞
- `.env` 檔案包含敏感資訊，請確保將其添加到 `.gitignore` 中

## 使用方法

### 基本使用

```python
from src.retriever import Document, ChromaVectorStore
from src.generator import generator
from main import rag_pipeline
from config import init_config

# 初始化配置
config = init_config()

# 創建向量儲存（啟用文本分塊）
vector_store = ChromaVectorStore(
    collection_name=config.collection_name,
    persist_directory=config.persist_directory,
    enable_chunking=True,  # 啟用文本分塊
    chunk_size=1000,       # 每個塊最大1000字符
    chunk_overlap=200      # 重疊200字符
)

# 添加文件到知識庫
documents = [
    Document("機器學習是人工智慧的一個分支。"),
    Document("深度學習使用神經網絡進行模式識別。"),
    Document("自然語言處理專注於計算機理解人類語言。")
]

vector_store.add_documents(documents)

# 執行 RAG pipeline
query = "什麼是機器學習？"
answer = rag_pipeline(query, vector_store)
print(answer)
```

### 命令行參數

你可以使用命令行參數來自定義配置：

```bash
# 使用預設配置
python main.py

# 自定義模型和參數
python main.py \
    --embedding-model "Qwen3-Embedding-4B" \
    --generator-model "Qwen2.5-7B-Instruct" \
    --retriever-top-k 30 \
    --temperature 0.5 \
    --verbose

# 使用自定義 API 端點
python main.py \
    --base-url "https://your-custom-endpoint.com/v1" \
    --collection-name "my_documents"

# 禁用 Reranker 階段（跳過重新排序）
python main.py --disable-reranker

# 啟用 Reranker 階段（預設行為）
python main.py --enable-reranker

### 個別組件使用

```python
from src.retriever import retriever
from src.reranker import reranker
from src.generator import generator

# 1. 檢索相關文件
retrieved_docs = retriever(query, vector_store, top_k=10)

# 2. 重新排序
reranked_docs = reranker(query, retrieved_docs, top_k=5)

# 3. 生成回答
answer = generator(query, reranked_docs)
```

### 自定義 API 設定

```python
from src.retriever import EmbeddingAPI
from src.reranker import RerankerAPI
from src.generator import GeneratorAPI

# 自定義 API 設定
embedding_api = EmbeddingAPI(
    api_key="your_api_key",
    base_url="https://your-api-endpoint.com/v1"
)

reranker_api = RerankerAPI(
    api_key="your_api_key",
    base_url="https://your-api-endpoint.com/v1"
)

generator_api = GeneratorAPI(
    api_key="your_api_key",
    base_url="https://your-api-endpoint.com/v1"
)
```

### 文本分塊設定

```python
from src.text_chunker import TextChunker, DocumentChunker

# 創建自定義文本分塊器
chunker = TextChunker(
    chunk_size=1000,      # 每個塊最大字符數
    chunk_overlap=200,    # 重疊字符數
    separator="\n\n",     # 自然分隔符
    min_chunk_size=100    # 最小塊大小
)

# 創建文檔分塊器
doc_chunker = DocumentChunker(chunker)

# 手動分塊文檔
chunked_docs = doc_chunker.chunk_documents(documents)
```

## API 端點

### Embedding API
- **端點**: `/embeddings`
- **模型**: `Qwen3-Embedding-4B`
- **功能**: 將文本轉換為向量表示

### Reranker API
- **端點**: `/rerank`
- **模型**: `bge-reranker-v2-m3`
- **功能**: 重新排序檢索結果

### Generator API
- **端點**: `/chat/completions`
- **模型**: `Qwen2.5-7B-Instruct`
- **功能**: 生成基於檢索內容的回答

## 檔案結構

```
rag-pipeline/
├── main.py                    # 主要 RAG pipeline 實現
├── config.py                  # 配置管理（命令行參數）
├── setup_env.py               # 環境變數設定腳本
├── example_usage.py           # 使用範例
├── test_api.py                # API 測試檔案
├── test_config.py             # 配置系統測試
├── test_evaluation.py         # 評估功能測試
├── test_chunking.py           # 文本分塊測試
├── test_generation_deepeval.py # DeepEval 測試
├── requirements.txt           # Python 依賴
├── README.md                 # 說明文件
└── src/                      # 核心模組
    ├── __init__.py           # 模組初始化
    ├── retriever.py          # 檢索模組
    ├── text_chunker.py       # 文本分塊模組
    ├── reranker.py           # 重新排序模組
    ├── generator.py          # 生成模組
    └── rag_deepeval.py       # 評估模組
```

## 核心類別

### Document
文件資料結構，包含內容、元數據和分數。

### VectorStore
簡單的向量儲存實現，支援文件添加和相似度搜尋。

### TextChunker
智能文本分塊器，支援：
- 可配置的塊大小和重疊
- 自然分隔符分割
- 句子邊界分割
- 語義完整性保持

### EmbeddingAPI
處理文本到向量的轉換。

### RerankerAPI
重新排序檢索結果以提高相關性。

### GeneratorAPI
基於檢索內容生成繁體中文回答，包含 system prompt 確保回答品質。

### 評估功能

專案包含基於 DeepEval 的評估功能，可以評估 RAG Pipeline 的表現：

```python
from src.rag_deepeval import evaluate_rag_pipeline

# 評估 RAG Pipeline
evaluation_results = evaluate_rag_pipeline(
    query="用戶查詢",
    actual_output="實際生成的回答",
    expected_output="期望的回答",
    retrieval_context=["檢索到的上下文"],
    metrics=["faithfulness", "answer_relevancy"]
)

# 查看評估結果
for metric_name, result in evaluation_results.items():
    print(f"{metric_name}: {result['score']}")
    print(f"原因: {result['reason']}")
```

支援的評估指標：
- **Faithfulness**: 評估回答是否忠實於檢索到的上下文
- **Answer Relevancy**: 評估回答與查詢的相關性
- **Contextual Precision**: 評估檢索上下文的精確度
- **Contextual Recall**: 評估檢索上下文的召回率
- **Contextual Relevancy**: 評估檢索上下文的相關性

### 評估模型配置

評估功能使用自定義的 LLM Judge 模型來進行評估判斷。你可以通過以下方式配置：

```bash
# 使用命令行參數指定評估模型
python main.py --llm-judge-model "gpt-oss-120b"

# 或在配置中設定
python main.py --llm-judge-model "gemma-3-12b-it"
```

預設的評估模型是 `gpt-oss-120b`，你可以根據需要選擇不同的模型來進行評估。

### 快速測試功能

為了方便快速測試 RAG Pipeline，我們提供了快速測試模式，只處理資料集的第一行：

```bash
# 快速測試模式（只處理第一行資料）
python main.py --dataset-name "your_dataset" --quick-test

# 完整測試模式（處理所有資料）
python main.py --dataset-name "your_dataset"
```

快速測試模式的特點：
- ⚡ **快速執行**：只處理資料集的第一行，節省時間
- 🔍 **詳細輸出**：顯示 context、query、expected answer 和 generated answer
- 🧪 **測試友好**：適合快速驗證 pipeline 功能
- 📊 **完整流程**：包含 retriever、reranker 和 generator 所有階段

## 執行範例

```bash
# 執行基本範例
python main.py

# 執行詳細使用範例
python example_usage.py

# 測試 API 連接
python test_api.py

# 測試配置系統
python test_config.py

# 測試中文回答功能
python test_chinese_response.py

# 測試評估功能
python test_evaluation.py

# 測試文本分塊功能
python test_chunking.py

# 使用自定義參數執行
python main.py --embedding-model "custom-model" --verbose

# 快速測試資料集功能
python test_quick_dataset.py

# 快速測試資料集（只處理第一行）
python main.py --dataset-name "your_dataset" --quick-test

# 完整測試資料集
python main.py --dataset-name "your_dataset"
```

## 配置選項

### 模型選擇
- **Embedding 模型**: `Qwen3-Embedding-4B`
- **Reranker 模型**: `bge-reranker-v2-m3`
- **Generator 模型**: `Qwen2.5-7B-Instruct`

### 參數調整
- **檢索數量**: `top_k` 參數控制檢索和重新排序的文件數量
- **生成參數**: `max_tokens` 和 `temperature` 控制生成品質

## 錯誤處理

程式包含完整的錯誤處理機制：
- API 連接錯誤
- 模型載入失敗
- 向量計算錯誤
- 文件處理異常

## 效能優化

- 使用餘弦相似度進行向量比較
- 支援批量文件處理
- 記憶體中的向量儲存
- 可配置的檢索參數

## 擴展功能

可以輕鬆擴展以下功能：
- 持久化向量儲存（如 Chroma、Pinecone）
- 更多相似度計算方法
- 自定義 prompt 模板
- 批次處理支援
- 快取機制

## 貢獻

歡迎提交 Issue 和 Pull Request 來改善這個專案。

## 授權

本專案採用 MIT 授權條款。 