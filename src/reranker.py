import requests
from typing import List, Dict, Any
from config import get_config
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
            # 使用 litellm 格式
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

        response = requests.post(url, headers=self.headers, json=data, timeout=30)
        if response.status_code == 200:
            raw = response.json()
            return self._parse_rerank_response(raw)
        else:
            raise Exception(f"Reranker API 錯誤: {response.status_code} - {response.text}")
    
    def _parse_rerank_response(self, raw: Any) -> List[Dict[str, Any]]:
        """解析重排序回應，統一格式為 [{"index": int, "score": float}]"""
        items = []
        
        # 提取結果列表
        payload_list = None
        if isinstance(raw, dict):
            # 嘗試常見的鍵名
            for key in ["results", "data", "ranking", "rankings"]:
                if isinstance(raw.get(key), list):
                    payload_list = raw[key]
                    break
            
            # 嘗試嵌套結構
            if payload_list is None:
                if isinstance(raw.get("results"), dict) and isinstance(raw["results"].get("items"), list):
                    payload_list = raw["results"]["items"]
                elif isinstance(raw.get("output"), dict) and isinstance(raw["output"].get("results"), list):
                    payload_list = raw["output"]["results"]
            
            # 嘗試 scores + indices 格式
            if payload_list is None and isinstance(raw.get("scores"), list):
                scores = raw.get("scores")
                indices = raw.get("indices") or list(range(len(scores)))
                payload_list = [{"index": idx, "score": sc} for idx, sc in zip(indices, scores)]
        
        elif isinstance(raw, list):
            payload_list = raw
        
        # 如果無法解析，返回空列表
        if not isinstance(payload_list, list):
            print(f"⚠️ Reranker 回傳格式無法解析: {type(raw)}")
            return []
        
        # 解析每個項目
        for i, item in enumerate(payload_list):
            if isinstance(item, dict):
                idx = item.get("index", i)
                score = self._extract_score(item)
                items.append({"index": idx, "score": score})
            elif isinstance(item, (int, float)):
                items.append({"index": i, "score": float(item)})
            else:
                items.append({"index": i, "score": 0.0})
        
        return items
    
    def _extract_score(self, item: Dict[str, Any]) -> float:
        """從項目中提取分數"""
        # 嘗試常見的分數鍵名
        for key in ["score", "relevance_score", "relevanceScore"]:
            if key in item and item[key] is not None:
                return float(item[key])
        
        # 嘗試使用 logit 作為分數
        if "logit" in item and item["logit"] is not None:
            return float(item["logit"])
        
        # 默認返回 0
        return 0.0

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
        
        print(f"🔍 Reranker 返回結果: {rerank_results}")
        
        # 創建文檔副本，避免修改原始對象
        doc_copies = []
        for i, doc in enumerate(documents):
            doc_copy = Document(
                content=doc.content,
                metadata=doc.metadata.copy() if doc.metadata else {},
                score=doc.score  # 保留原始分數
            )
            doc_copies.append(doc_copy)
        
        # 更新文件分數（依回傳 index 對應）
        for result in rerank_results:
            idx = result.get("index")
            score = result.get("score", 0.0)
            if isinstance(idx, int) and 0 <= idx < len(doc_copies):
                doc_copies[idx].score = score
                print(f"🔍 更新文檔 {idx} 分數為: {score:.6f}")
            else:
                print(f"⚠️ 無效的索引 {idx} 或分數 {score}")
        
        # 按分數排序
        doc_copies.sort(key=lambda x: x.score, reverse=True)
        
        print(f"✅ 重新排序完成，返回前 {min(top_k, len(doc_copies))} 個文件")
        return doc_copies[:top_k]
    
    except Exception as e:
        print(f"⚠️ Reranker API 錯誤，使用原始排序: {e}")
        # 返回原始文檔的副本，保留原始分數
        return [Document(
            content=doc.content, 
            metadata=doc.metadata.copy() if doc.metadata else {}, 
            score=doc.score
        ) for doc in documents[:top_k]]
