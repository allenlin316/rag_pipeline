#!/usr/bin/env python3
"""
測試中文回答功能的腳本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import Document, ChromaVectorStore, rag_pipeline, GeneratorAPI
from config import init_config

def test_chinese_response():
    """測試中文回答功能"""
    print("🧪 測試中文回答功能")
    print("=" * 60)
    
    # 初始化配置
    config = init_config()
    
    # 創建向量儲存
    vector_store = ChromaVectorStore(
        collection_name="test_chinese",
        persist_directory="./test_chinese_chroma_db"
    )
    
    # 添加中文測試文件
    test_documents = [
        Document("人工智慧（AI）是計算機科學的一個分支，致力於創建能夠執行通常需要人類智能的任務的系統。"),
        Document("機器學習是人工智慧的一個子集，它使計算機能夠從數據中學習而無需明確編程。"),
        Document("深度學習是機器學習的一個分支，使用多層神經網絡來處理複雜的模式識別任務。"),
        Document("自然語言處理（NLP）是人工智慧的一個領域，專注於計算機理解和生成人類語言。"),
        Document("計算機視覺是人工智慧的一個分支，使計算機能夠理解和解釋視覺信息。")
    ]
    
    print("📚 添加中文測試文件...")
    vector_store.add_documents(test_documents)
    
    # 測試查詢
    test_queries = [
        "什麼是機器學習？",
        "請解釋深度學習的概念",
        "自然語言處理有什麼應用？",
        "計算機視覺能做什麼？"
    ]
    
    print(f"\n🔍 測試 {len(test_queries)} 個查詢...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n" + "="*40)
        print(f"❓ 查詢 {i}: {query}")
        print("="*40)
        
        answer = rag_pipeline(query, vector_store)
        print(f"💬 回答: {answer}")
        
        # 檢查是否包含中文字符
        chinese_chars = sum(1 for char in answer if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > 0:
            print(f"✅ 回答包含 {chinese_chars} 個中文字符")
        else:
            print("⚠️ 回答中沒有檢測到中文字符")
    
    # 測試 Generator API 的 system prompt
    print(f"\n" + "="*40)
    print("🔍 測試 Generator API 的 system prompt")
    print("="*40)
    
    generator_api = GeneratorAPI()
    test_prompt = "請用繁體中文介紹人工智慧的基本概念。"
    
    try:
        answer = generator_api.generate(test_prompt)
        print(f"❓ 測試 prompt: {test_prompt}")
        print(f"💬 回答: {answer}")
        
        # 檢查是否包含中文字符
        chinese_chars = sum(1 for char in answer if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > 0:
            print(f"✅ 回答包含 {chinese_chars} 個中文字符")
        else:
            print("⚠️ 回答中沒有檢測到中文字符")
            
    except Exception as e:
        print(f"❌ Generator API 測試失敗: {e}")
    
    # 清理測試數據
    print("\n🧹 清理測試數據...")
    try:
        vector_store.delete_collection()
        import shutil
        if os.path.exists("./test_chinese_chroma_db"):
            shutil.rmtree("./test_chinese_chroma_db")
        print("✅ 測試數據清理完成")
    except Exception as e:
        print(f"⚠️ 清理測試數據時發生錯誤: {e}")
    
    print("\n🎉 中文回答功能測試完成！")

if __name__ == "__main__":
    test_chinese_response() 