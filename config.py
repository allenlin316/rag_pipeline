import os
import argparse
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieval-Augmented Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # API 設定 - API Key 只能從環境變數讀取
    parser.add_argument(
        "--retrieval-reranker-api-key",
        type=str,
        help="API Key for authentication (建議使用 .env 檔案設定)"
    )
    parser.add_argument(
        "--generator-api-key",
        type=str,
        help="API Key for authentication (建議使用 .env 檔案設定)"
    )
    parser.add_argument(
        "--retriever-base-url",
        type=str,
        default="https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai/v1",
        help="Base URL for API endpoints"
    )
    parser.add_argument(
        "--reranker-base-url",
        type=str,
        default="http://192.168.80.11:8100/v1",
        help="Base URL for Reranker API endpoints"
    )
    parser.add_argument(
        "--generator-base-url",
        type=str,
        default="https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai/v1",
        help="Base URL for Generator API endpoints"
    )
    
    # 模型設定
    parser.add_argument(
        "--embedding-model", "-c",
        dest="embedding_model",
        type=str,
        default="llama-3.2-nv-embedqa-1b-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        help="Reranker model name"
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="gemma-3-27b-it",
        help="Generator model name"
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default="Qwen3-30B-A3B",
        help="LLM Judge model name"
    )
    
    # Retriever 參數
    parser.add_argument(
        "--retriever-top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve in retriever stage"
    )
    
    # Reranker 參數
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        default=True,
        help="Enable reranker stage (default: True)"
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranker stage"
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=5,
        help="Number of documents to return after reranking"
    )
    
    # Generator 參數
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for text generation (0.0-1.0)"
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        default=False,
        help="Skip special tokens in generation (useful for Ace1-24B-Security model)"
    )
    
    # Chroma 向量儲存參數
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rag_documents",
        help="Chroma collection name"
    )
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="./chroma_db",
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete existing Chroma collection before indexing (start fresh)"
    )
    # 評估參數
    parser.add_argument(
        "--enable-eval",
        action="store_true",
        default=True,
        help="Enable running evaluation after generation (default: True)"
    )
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Disable evaluation"
    )
    parser.add_argument(
        "--ground-truth-context-file",
        type=str,
        default=None,
        help="Path to a JSON file containing ground truth contexts as a list of strings"
    )
    parser.add_argument(
        "--expected-output-text",
        type=str,
        default="",
        help="Optional expected output text for evaluation"
    )
    # 評估結果儲存參數
    parser.add_argument(
        "--enable-metrics-logging",
        action="store_true",
        default=True,
        help="Enable appending evaluation metrics to a CSV file"
    )
    parser.add_argument(
        "--disable-metrics-logging",
        action="store_true",
        help="Disable metrics CSV logging"
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=str,
        default="metrics_logs.csv",
        help="Path to the CSV file for saving evaluation metrics"
    )
    # Pipeline 結果儲存參數（input、前五個 rerank 文件、generation 結果）
    parser.add_argument(
        "--enable-results-logging",
        action="store_true",
        default=True,
        help="Enable appending query, top-5 reranked docs, and generation output to a JSONL file"
    )
    parser.add_argument(
        "--disable-results-logging",
        action="store_true",
        help="Disable results JSONL logging"
    )
    parser.add_argument(
        "--results-jsonl-path",
        type=str,
        default="results_logs.jsonl",
        help="Path to the JSONL file for saving pipeline results"
    )
    # 文本分塊參數
    parser.add_argument(
        "--enable-chunking",
        action="store_true",
        default=True,
        help="Enable text chunking (default: True)"
    )
    parser.add_argument(
        "--disable-chunking",
        action="store_true",
        help="Disable text chunking"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Text chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters"
    )
    
    # 其他參數
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with sample data"
    )
    # Hugging Face token for gated datasets
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for gated datasets"
    )
    # Dataset 批次評估選項
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset identifier (e.g., 'user/repo' or 'squad')"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Optional dataset split (e.g., 'train'|'validation'|'test'). If not provided, use all available splits."
    )
    parser.add_argument(
        "--context-field",
        type=str,
        default="context",
        help="Field name for context in dataset"
    )
    parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Field name for question (user query) in dataset"
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="answer",
        help="Field name for ground truth answer in dataset"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: only process the first row of the dataset"
    )
    
    args = parser.parse_args()
    
    # 處理 reranker 開關邏輯
    if args.disable_reranker:
        args.enable_reranker = False
    # 處理 chunking 開關邏輯
    if args.disable_chunking:
        args.enable_chunking = False
    # 處理 eval 開關邏輯
    if args.disable_eval:
        args.enable_eval = False
    # 處理 metrics logging 開關邏輯
    if args.disable_metrics_logging:
        args.enable_metrics_logging = False
    # 處理 results logging 開關邏輯
    if args.disable_results_logging:
        args.enable_results_logging = False
    
    # 確保 API Key 優先從環境變數讀取
    if not args.retrieval_reranker_api_key or not args.generator_api_key:
        #args.api_key = os.getenv("API_KEY")
        args.retrieval_reranker_api_key = os.getenv("RETRIEVAL_RERANKER_API_KEY")
        args.generator_api_key = os.getenv("GENERATOR_API_KEY")
        if not args.retrieval_reranker_api_key or not args.generator_api_key:
            print("⚠️  警告: 未設定 API_KEY 環境變數")
            print("   請在 .env 檔案中設定 API_KEY=your_api_key_here")
            args.api_key = "none"  # 預設值

    # 讀取 HF token（支援兩種常見環境變數名稱）
    if not args.hf_token:
        args.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    
    return args

# 全域變數來儲存參數
ARGS = None

def init_config():
    """初始化配置，解析命令行參數"""
    global ARGS
    ARGS = parse_arguments()
    return ARGS

def get_config():
    """獲取當前配置"""
    global ARGS
    if ARGS is None:
        ARGS = parse_arguments()
    return ARGS 