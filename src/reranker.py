import requests
from typing import List, Dict, Any
from config import get_config

# 定義 Document 類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .retriever import Document

class RerankerAPI:
    """Reranker API 客戶端

    - 本地/內網服務: 使用 /ranking 與 {query: {text}, passages: [{text}]}
    - litellm-ekkks8gsocw.dgx-coolify.apmic.ai: 使用 /v1/rerank 與 {query, documents}
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        config = get_config()
        self.api_key = config.generator_api_key
        self.base_url = base_url or config.reranker_base_url
        self.model = model or config.reranker_model
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        if self.api_key and self.api_key != "none":
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _use_litellm_format(self) -> bool:
        """當 base_url 指向 litellm 服務時，改用 /v1/rerank 與簡化 payload。"""
        return "litellm-ekkks8gsocw.dgx-coolify.apmic.ai" in (self.base_url or "")

    def rerank(self, query: str, documents: List[str], model: str = None, truncate: str = "END") -> List[Dict[str, Any]]:
        """重新排序文件，依 base_url 選擇不同的 API 格式"""
        if self._use_litellm_format():
            # 使用 quick_test.py 相同格式
            data = {
                "model": model or self.model,
                "query": query,
                "documents": documents,
            }
            url = f"{self.base_url}/v1/rerank"
        else:
            # 本地/內網 reranker 服務
            data = {
                "model": model or self.model,
                "query": {"text": query},
                "passages": [{"text": doc} for doc in documents],
                "truncate": truncate
            }
            url = f"{self.base_url}/ranking"

        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            raw = response.json()
            # 將不同格式統一轉為 [{"index": int, "score": float}]
            items = []
            payload_list = None
            if isinstance(raw, dict):
                if isinstance(raw.get("results"), list):
                    payload_list = raw["results"]
                elif isinstance(raw.get("data"), list):
                    payload_list = raw["data"]
            elif isinstance(raw, list):
                payload_list = raw

            if not isinstance(payload_list, list):
                return []

            for i, item in enumerate(payload_list):
                if isinstance(item, dict):
                    idx = item.get("index", i)
                    score = item.get("score")
                    if score is None:
                        score = item.get("relevance_score")
                    # 仍找不到分數，嘗試常見鍵
                    if score is None:
                        score = item.get("relevanceScore")
                    if score is None:
                        # 無法辨識則設為 0
                        score = 0.0
                    items.append({"index": idx, "score": float(score) if isinstance(score, (int, float, str)) else 0.0})
                elif isinstance(item, (int, float)):
                    items.append({"index": i, "score": float(item)})
                else:
                    items.append({"index": i, "score": 0.0})
            return items
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
        
        # 更新文件分數（依回傳 index 對應）
        for i, doc in enumerate(documents):
            # 預設為 0，避免殘留舊分數
            try:
                doc.score = 0.0
            except Exception:
                pass
        for result in rerank_results:
            idx = result.get("index")
            score = result.get("score", 0.0)
            if isinstance(idx, int) and 0 <= idx < len(documents):
                documents[idx].score = score
        
        # 按分數排序
        documents.sort(key=lambda x: x.score, reverse=True)
        
        print(f"✅ 重新排序完成，返回前 {min(top_k, len(documents))} 個文件")
        return documents[:top_k]
    
    except Exception as e:
        print(f"⚠️ Reranker API 錯誤，使用原始排序: {e}")
        return documents[:top_k]
