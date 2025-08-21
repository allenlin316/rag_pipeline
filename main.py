import os
import json
from typing import List, Dict, Any, Optional, get_origin, get_args
import csv
from datetime import datetime, timezone, timedelta
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
from datasets import load_dataset
from pydantic import BaseModel

def rag_pipeline(query: str, vector_store: ChromaVectorStore, expected_output_text: Optional[str] = None) -> str:
    """
    完整的 RAG Pipeline
    
    Args:
        query: 查詢文本
        vector_store: 向量儲存實例
        expected_output_text: 當有資料集時，直接傳入 answer 作為評估的 expected output
    """
    config = get_config()
    print("🚀 開始 RAG Pipeline")
    print("=" * 50)
    
    # 1. Retriever 階段
    retrieved_docs = retriever(query, vector_store, top_k=config.retriever_top_k)
    
    # 2. Reranker 階段 (可選)
    if config.enable_reranker:
        print(f"🔄 啟用 {config.reranker_model} Reranker 階段")
        reranked_docs = reranker(query, retrieved_docs, top_k=config.reranker_top_k)
    else:
        print("⏭️  跳過 Reranker 階段")
        reranked_docs = retrieved_docs[:config.reranker_top_k]
    
    # 3. Generator 階段
    answer = generator(query, reranked_docs)

    # 3.5 儲存 input、檢索結果（pre-rerank 全部）、前 top-k rerank 文件、與 generation 結果（JSONL）
    try:
        if getattr(config, "enable_results_logging", False):
            results_path = getattr(config, "results_jsonl_path", "results_logs.jsonl")
            top_k = getattr(config, "reranker_top_k", 3)
            top_docs = reranked_docs[:top_k]
            # 序列化：檢索結果（pre-rerank 全部）
            retrieved_serialized_docs = []
            for idx, doc in enumerate(retrieved_docs, start=1):
                try:
                    retrieved_serialized_docs.append({
                        "retrieval_rank": idx,
                        "score": getattr(doc, "score", None),
                        "content": getattr(doc, "content", None),
                        "metadata": getattr(doc, "metadata", None),
                    })
                except Exception:
                    retrieved_serialized_docs.append({
                        "retrieval_rank": idx,
                        "score": None,
                        "content": None,
                        "metadata": None,
                    })
            # 產生 UTC+8 時間字串
            tz_utc8 = timezone(timedelta(hours=8))
            timestamp_utc8 = datetime.now(tz_utc8).strftime("%Y-%m-%d %H:%M:%S %z")
            # 序列化文件
            serialized_docs = []
            for idx, doc in enumerate(top_docs, start=1):
                try:
                    serialized_docs.append({
                        "rank": idx,
                        "score": getattr(doc, "score", None),
                        "content": getattr(doc, "content", None),
                        "metadata": getattr(doc, "metadata", None),
                    })
                except Exception:
                    serialized_docs.append({
                        "rank": idx,
                        "score": None,
                        "content": None,
                        "metadata": None,
                    })
            record = {
                "timestamp_utc8": timestamp_utc8,
                "query": query,
                "retrieved_docs": retrieved_serialized_docs,
                "top5_reranked_docs": serialized_docs,
                "answer": answer,
            }
            with open(results_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"💾 已儲存查詢與生成結果到 {results_path}")
    except Exception as e:
        print(f"⚠️ 儲存結果到 JSONL 失敗: {e}")

    # 4. 評估（可選）
    if getattr(config, "enable_eval", True):
        print("\n🧪 執行評估...")
        # 準備 retrieval contexts（取回傳文件的 content 作為上下文）
        retrieval_context = [d.content for d in reranked_docs]
        # 優先使用函數參數，再回退 config，最後回退空字串
        expected_output = str((expected_output_text if expected_output_text is not None else getattr(config, "expected_output_text", "")) or "")
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
                llm_judge_model=getattr(config, 'llm_judge_model', 'gpt-oss-120b')
            )
            print("\n📈 評估結果：")
            for k, v in results.items():
                print(f"  {k}: {v.get('score')} | {v.get('reason')}")

            # 可選：儲存到 CSV
            if getattr(config, "enable_metrics_logging", False):
                try:
                    # 蒐集模型資訊
                    embedding_model = getattr(config, "embedding_model", None)
                    reranker_model = getattr(config, "reranker_model", None)
                    generator_model = getattr(config, "generator_model", None)

                    # 蒐集指標分數
                    faithfulness = (results.get("faithfulness") or {}).get("score")
                    answer_relevancy = (results.get("answer_relevancy") or {}).get("score")
                    contextual_precision = (results.get("contextual_precision") or {}).get("score")
                    contextual_recall = (results.get("contextual_recall") or {}).get("score")
                    contextual_relevancy = (results.get("contextual_relevancy") or {}).get("score")

                    # 蒐集指標原因
                    faithfulness_reason = (results.get("faithfulness") or {}).get("reason")
                    answer_relevancy_reason = (results.get("answer_relevancy") or {}).get("reason")
                    contextual_precision_reason = (results.get("contextual_precision") or {}).get("reason")
                    contextual_recall_reason = (results.get("contextual_recall") or {}).get("reason")
                    contextual_relevancy_reason = (results.get("contextual_relevancy") or {}).get("reason")

                    csv_path = getattr(config, "metrics_csv_path", "metrics_logs.csv")
                    # 首次建立或空檔都寫入表頭
                    try:
                        file_empty = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
                    except Exception:
                        file_empty = True

                    # 建立/附加 CSV
                    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "timestamp_utc8",
                                "query",
                                "embedding_model",
                                "reranker_model",
                                "generator_model",
                                "faithfulness",
                                "faithfulness_reason",
                                "answer_relevancy",
                                "answer_relevancy_reason",
                                "contextual_precision",
                                "contextual_precision_reason",
                                "contextual_recall",
                                "contextual_recall_reason",
                                "contextual_relevancy",
                                "contextual_relevancy_reason",
                            ],
                        )
                        if file_empty:
                            writer.writeheader()
                        # 產生 UTC+8 時間字串（例如 2025-08-19 14:30:00 +0800）
                        tz_utc8 = timezone(timedelta(hours=8))
                        timestamp_utc8 = datetime.now(tz_utc8).strftime("%Y-%m-%d %H:%M:%S %z")
                        writer.writerow({
                            "timestamp_utc8": timestamp_utc8,
                            "query": query,
                            "embedding_model": embedding_model,
                            "reranker_model": reranker_model,
                            "generator_model": generator_model,
                            "faithfulness": faithfulness,
                            "faithfulness_reason": faithfulness_reason,
                            "answer_relevancy": answer_relevancy,
                            "answer_relevancy_reason": answer_relevancy_reason,
                            "contextual_precision": contextual_precision,
                            "contextual_precision_reason": contextual_precision_reason,
                            "contextual_recall": contextual_recall,
                            "contextual_recall_reason": contextual_recall_reason,
                            "contextual_relevancy": contextual_relevancy,
                            "contextual_relevancy_reason": contextual_relevancy_reason,
                        })
                    print(f"💾 已儲存評估結果到 {csv_path}")
                except Exception as e:
                    print(f"⚠️ 儲存評估結果到 CSV 失敗: {e}")
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

    # 若提供 dataset，改為批次流程
    if getattr(config, "dataset_name", None):
        chosen_split = getattr(config, "dataset_split", None)
        if chosen_split:
            ds = load_dataset(config.dataset_name, split=chosen_split, token=getattr(config, "hf_token", None))
            datasets_to_run = [(chosen_split, ds)]
        else:
            ds_all = load_dataset(config.dataset_name, token=getattr(config, "hf_token", None))
            # 收集所有可用 splits
            if hasattr(ds_all, "keys"):
                datasets_to_run = [(name, ds_all[name]) for name in ds_all.keys()]
            else:
                datasets_to_run = [("unspecified", ds_all)]
        print("📦 使用以下 splits 進行測試: " + ", ".join([name for name, _ in datasets_to_run]))
        ctx_field = getattr(config, "context_field", "context")
        q_field = getattr(config, "question_field", "question")
        a_field = getattr(config, "answer_field", "answer")

        for split_name, split_ds in datasets_to_run:
            print(f"\n===== Split: {split_name} =====")
            
            # 為每個 split 創建新的向量儲存實例
            vector_store = ChromaVectorStore(
                collection_name=f"{config.collection_name}_{split_name}",
                persist_directory=config.persist_directory,
                enable_chunking=config.enable_chunking,
                chunk_size=getattr(config, 'chunk_size', 512),
                chunk_overlap=getattr(config, 'chunk_overlap', 200),
                reset=getattr(config, 'reset_collection', False)
            )

            # 檢查是否為快速測試模式
            if getattr(config, "quick_test", False):
                print("⚡ 快速測試模式：只處理第一行資料")
                split_ds = split_ds.select(range(15))  # 只取前 15 行
            
            # 將該 split 的所有 context 加入知識庫
            print("📚 將 contexts 加入 Chroma...")
            docs = []
            for i, row in enumerate(split_ds):
                context_text = row.get(ctx_field)
                if not context_text:
                    continue
                docs.append(Document(str(context_text)))
            if docs:
                vector_store.add_documents(docs)

            # 逐筆執行
            print("🚀 逐筆執行 RAG 與評估（append CSV）...")
            for idx, row in enumerate(split_ds):
                query = str(row.get(q_field, "")).strip()
                gt_answer = str(row.get(a_field, "")).strip()
                if not query:
                    continue
                print(f"\n----- Case {idx+1} -----")
                print(f"Q: {query}")
                print(f"Context: {row.get(ctx_field, '')[:100]}...")  # 顯示前100個字符的context
                ans = rag_pipeline(query, vector_store, expected_output_text=gt_answer)
                print(f"A: {ans}")
                print(f"Expected: {gt_answer}")
                
                # 快速測試模式下只處理第一行就結束
                if getattr(config, "quick_test", False):
                    print("⚡ 快速測試完成，只處理了第一行資料")
                    break

        print("\n✅ Dataset 批次流程完成")
    else:
        # 單例示範模式 - 創建 Chroma 向量儲存
        vector_store = ChromaVectorStore(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            enable_chunking=config.enable_chunking,
            chunk_size=getattr(config, 'chunk_size', 512),
            chunk_overlap=getattr(config, 'chunk_overlap', 200),
            reset=getattr(config, 'reset_collection', False)
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
