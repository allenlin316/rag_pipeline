import requests
from typing import List, Dict, Any
from config import get_config

# 定義 Document 類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .retriever import Document

class RerankerAPI:
    """Reranker API 客戶端 (新版，呼叫 /v1/ranking)"""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        config = get_config()
        self.api_key = config.retrieval_reranker_api_key
        self.base_url = base_url or config.reranker_base_url
        self.model = model or config.reranker_model
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        if self.api_key and self.api_key != "none":
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def rerank(self, query: str, documents: List[str], model: str = None, truncate: str = "END") -> List[Dict[str, Any]]:
        """重新排序文件 (新版 API)"""
        data = {
            "model": model or self.model,
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents],
            "truncate": truncate
        }
        response = requests.post(
            f"{self.base_url}/ranking",
            headers=self.headers,
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            # 假設 response['results'] 是排序後的 passages，帶有分數與 index
            return result.get("results", [])
        else:
            raise Exception(f"Reranker API 錯誤: {response.status_code} - {response.text}")

def reranker(query: str, documents: List["Document"], top_k: int = 5) -> List["Document"]:
    """
    Reranker 函數：重新排序檢索到的文件
    
    Args:
        query: 查詢文本
        documents: 檢索到的文件列表
        top_k: 重新排序後返回的文件數量
    
    Returns:
        重新排序後的文件列表
    """
    print(f"🔄 重新排序 {len(documents)} 個文件")
    
    if not documents:
        return []
    
    reranker_api = RerankerAPI()
    doc_contents = [doc.content for doc in documents]
    
    try:
        rerank_results = reranker_api.rerank(query, doc_contents)
        
        # 更新文件分數和排序
        for i, result in enumerate(rerank_results):
            documents[i].score = result["score"]
        
        # 按分數排序
        documents.sort(key=lambda x: x.score, reverse=True)
        
        print(f"✅ 重新排序完成，返回前 {min(top_k, len(documents))} 個文件")
        return documents[:top_k]
    
    except Exception as e:
        print(f"⚠️ Reranker API 錯誤，使用原始排序: {e}")
        return documents[:top_k]
