import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from config import get_config, init_config

# 導入新的模組
from src.retriever import Document, ChromaVectorStore, retriever
from src.reranker import reranker
from src.generator import generator
from src.rag_deepeval import (
    evaluate_rag_pipeline,
    CustomLLMJudge,
)

def rag_pipeline(query: str, vector_store: ChromaVectorStore) -> str:
    """
    完整的 RAG Pipeline
    
    Args:
        query: 查詢文本
        vector_store: 向量儲存實例
    """
    config = get_config()
    print("🚀 開始 RAG Pipeline")
    print("=" * 50)
    
    # 1. Retriever 階段
    retrieved_docs = retriever(query, vector_store, top_k=config.retriever_top_k)
    
    # 2. Reranker 階段 (可選)
    if config.enable_reranker:
        print("🔄 啟用 Reranker 階段")
        reranked_docs = reranker(query, retrieved_docs, top_k=config.reranker_top_k)
    else:
        print("⏭️  跳過 Reranker 階段")
        reranked_docs = retrieved_docs[:config.reranker_top_k]
    
    # 3. Generator 階段
    answer = generator(query, reranked_docs)

    # 4. 評估（可選）
    if getattr(config, "enable_eval", False):
        print("\n🧪 執行評估...")
        # 準備 retrieval contexts（取回傳文件的 content 作為上下文）
        retrieval_context = [d.content for d in reranked_docs]
        expected_output = getattr(config, "expected_output_text", None) or ""
        # 指標：retrieval eval（precision、recall、relevancy），generator eval（faithfulness、answer relevancy）
        metrics = [
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "faithfulness",
            "answer_relevancy",
        ]
        try:
            results = evaluate_rag_pipeline(
                query=query,
                actual_output=answer,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
                metrics=metrics,
            )
            print("\n📈 評估結果：")
            for k, v in results.items():
                print(f"  {k}: {v.get('score')} | {v.get('reason')}")
        except Exception as e:
            print(f"⚠️ 評估失敗: {e}")
    
    print("=" * 50)
    print("✅ RAG Pipeline 完成")
    
    return answer

# 示例使用
if __name__ == "__main__":
    # 初始化配置
    config = init_config()
    
    # 顯示 Reranker 設定
    print(f"🔄 Reranker 設定: {'啟用' if config.enable_reranker else '禁用'}")
    
    # 創建 Chroma 向量儲存
    vector_store = ChromaVectorStore(
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
        enable_chunking=config.enable_chunking,
        chunk_size=getattr(config, 'chunk_size', 512),
        chunk_overlap=getattr(config, 'chunk_overlap', 200)
    )
    
    # 顯示集合資訊
    info = vector_store.get_collection_info()
    print(f"📊 集合資訊: {info}")
    
    # 示例文件
    sample_documents = [
        Document("人工智慧（AI）是計算機科學的一個分支，致力於創建能夠執行通常需要人類智能的任務的系統。"),
        Document("機器學習是人工智慧的一個子集，它使計算機能夠從數據中學習而無需明確編程。"),
        Document("深度學習是機器學習的一個分支，使用多層神經網絡來處理複雜的模式識別任務。"),
        Document("自然語言處理（NLP）是人工智慧的一個領域，專注於計算機理解和生成人類語言。"),
        Document("計算機視覺是人工智慧的一個分支，使計算機能夠理解和解釋視覺信息。")
    ]
    
    print("📚 添加示例文件到 Chroma...")
    vector_store.add_documents(sample_documents)
    
    # 顯示更新後的集合資訊
    info = vector_store.get_collection_info()
    print(f"📊 更新後集合資訊: {info}")
    
    # 測試查詢
    test_query = "什麼是機器學習？"
    print(f"\n❓ 測試查詢: {test_query}")
    
    # 執行 RAG pipeline
    answer = rag_pipeline(test_query, vector_store)
    print(f"\n💬 回答: {answer}")
    
    # 顯示使用提示
    print(f"\n💡 使用提示:")
    print(f"   - 啟用 Reranker: python main.py --enable-reranker")
    print(f"   - 禁用 Reranker: python main.py --disable-reranker")
    print(f"   - 測試中文回答: python test_chinese_response.py")
