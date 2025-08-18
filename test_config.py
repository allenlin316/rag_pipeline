#!/usr/bin/env python3
"""
測試配置系統的腳本
"""

from config import init_config, get_config
import sys

def test_config():
    """測試配置系統"""
    print("🧪 測試配置系統")
    print("=" * 50)
    
    # 初始化配置
    config = init_config()
    
    print("📋 當前配置:")
    print(f"  API Key: {config.api_key[:10]}..." if len(config.api_key) > 10 else f"  API Key: {config.api_key}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Reranker Model: {config.reranker_model}")
    print(f"  Generator Model: {config.generator_model}")
    print(f"  Retriever Top-K: {config.retriever_top_k}")
    print(f"  Reranker Top-K: {config.reranker_top_k}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Collection Name: {config.collection_name}")
    print(f"  Persist Directory: {config.persist_directory}")
    print(f"  Verbose: {config.verbose}")
    print(f"  Test Mode: {config.test_mode}")
    
    # 測試獲取配置
    config2 = get_config()
    print(f"\n✅ 配置一致性檢查: {config == config2}")
    
    return config

if __name__ == "__main__":
    # 測試不同的命令行參數
    print("🚀 測試配置系統")
    
    # 測試預設配置
    print("\n1️⃣ 測試預設配置:")
    sys.argv = ["test_config.py"]
    config1 = test_config()
    
    # 測試自定義配置
    print("\n2️⃣ 測試自定義配置:")
    sys.argv = [
        "test_config.py",
        "--embedding-model", "custom-embedding",
        "--generator-model", "custom-generator",
        "--retriever-top-k", "30",
        "--temperature", "0.5",
        "--verbose"
    ]
    config2 = test_config()
    
    print("\n" + "=" * 50)
    print("✅ 配置系統測試完成")
    print("=" * 50) 