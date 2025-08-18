import requests
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from dataclasses import dataclass
from config import get_config
from .text_chunker import DocumentChunker, TextChunker

@dataclass
class Document:
    """文件資料結構"""
    content: str
    metadata: Dict[str, Any] = None
    score: float = 0.0

class EmbeddingAPI:
    """Embedding API 客戶端"""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        config = get_config()
        self.api_key = config.retrieval_reranker_api_key
        self.base_url = base_url or config.retriever_base_url
        self.model = model or config.embedding_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """獲取文本的 embedding"""
        if not text or not text.strip():
            raise Exception("文本內容為空")
        
        data = {
            "input": text,
            "model": model or self.model
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=data,
                timeout=30  # 添加超時
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0]["embedding"]
                    if embedding and len(embedding) > 0:
                        return embedding
                    else:
                        raise Exception("API 返回的 embedding 為空")
                else:
                    raise Exception("API 返回的數據格式不正確")
            else:
                raise Exception(f"API 請求失敗: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"網絡請求錯誤: {e}")
        except Exception as e:
            raise Exception(f"獲取 embedding 時發生錯誤: {e}")

class ChromaVectorStore:
    """使用 Chroma 的向量儲存"""
    def __init__(
        self, 
        collection_name: str = "rag_documents", 
        persist_directory: str = "./chroma_db",
        enable_chunking: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 200
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.enable_chunking = enable_chunking
        
        # 初始化 Chroma 客戶端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 獲取或創建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ 載入現有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG Pipeline 文件集合"}
            )
            print(f"✅ 創建新集合: {collection_name}")
        
        # Embedding API 客戶端
        self.embedding_api = EmbeddingAPI()
        
        # 文本分塊器
        if self.enable_chunking:
            self.chunker = DocumentChunker(
                TextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            )
            print(f"✅ 啟用文本分塊 (chunk_size={chunk_size}, overlap={chunk_overlap})")
        else:
            self.chunker = None
            print("⏭️ 禁用文本分塊")
    
    def add_documents(self, documents: List[Document]):
        """添加文件到 Chroma 向量儲存"""
        print(f"📚 添加 {len(documents)} 個文件到 Chroma...")
        
        # 如果啟用分塊，先進行文本分塊
        if self.enable_chunking and self.chunker:
            print("🔄 進行文本分塊...")
            chunked_documents = self.chunker.chunk_documents(documents)
            print(f"📄 分塊後得到 {len(chunked_documents)} 個文本塊")
            # 若分塊結果為空，回退為原始文檔
            documents_to_add = chunked_documents or documents
        else:
            documents_to_add = documents
        
        # 準備數據
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        # 先嘗試獲取一個 embedding 來確定維度
        embedding_dimension = None
        try:
            test_embedding = self.embedding_api.get_embedding("test")
            embedding_dimension = len(test_embedding)
            print(f"✅ 確認 embedding 維度: {embedding_dimension}")
        except Exception as e:
            print(f"⚠️ 無法確定 embedding 維度: {e}")
            embedding_dimension = 1536  # 預設維度
        
        for i, doc in enumerate(documents_to_add):
            # 生成唯一 ID
            doc_id = f"doc_{i}_{hash(doc.content) % 10000}"
            ids.append(doc_id)
            
            # 文本內容
            texts.append(doc.content)
            
            # 元數據
            metadata = doc.metadata or {}
            metadata.update({
                "source": "rag_pipeline",
                "content_length": len(doc.content),
                "is_chunked": self.enable_chunking
            })
            metadatas.append(metadata)
            
            # 獲取 embedding
            try:
                embedding = self.embedding_api.get_embedding(doc.content)
                if embedding and len(embedding) > 0:
                    embeddings.append(embedding)
                else:
                    raise Exception("獲取到的 embedding 為空")
            except Exception as e:
                print(f"⚠️ 獲取 embedding 失敗 (文檔 {i+1}): {e}")
                # 如果 API 失敗，使用零向量作為備用
                zero_embedding = [0.0] * embedding_dimension
                embeddings.append(zero_embedding)
        
        # 若沒有可添加的文檔，直接返回
        if not ids and len(documents_to_add) == 0:
            print("⚠️ 沒有可添加的文件，跳過")
            return

        # 批量添加到 Chroma
        try:
            if not ids or not embeddings:
                print("⚠️ 空的 ids 或 embeddings，跳過添加")
                return
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"✅ 成功添加 {len(documents_to_add)} 個文件到 Chroma")
        except Exception as e:
            print(f"❌ 添加文件到 Chroma 失敗: {e}")
            raise
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """搜尋相似文件"""
        try:
            # 獲取查詢的 embedding
            query_embedding = self.embedding_api.get_embedding(query)
            
            # 在 Chroma 中搜尋
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 轉換為 Document 物件
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # 將距離轉換為相似度分數 (Chroma 使用歐幾里得距離)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    doc = Document(
                        content=doc_text,
                        metadata=metadata,
                        score=similarity_score
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Chroma 搜尋失敗: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """獲取集合資訊"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"❌ 獲取集合資訊失敗: {e}")
            return {}
    
    def delete_collection(self):
        """刪除集合"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✅ 刪除集合: {self.collection_name}")
        except Exception as e:
            print(f"❌ 刪除集合失敗: {e}")

def retriever(query: str, vector_store: ChromaVectorStore, top_k: int = 20) -> List[Document]:
    """
    Retriever 函數：從向量儲存中檢索相關文件
    
    Args:
        query: 查詢文本
        vector_store: 向量儲存實例
        top_k: 返回的文件數量
    
    Returns:
        相關文件列表
    """
    print(f"🔍 檢索查詢: {query}")
    documents = vector_store.search(query, top_k=top_k)
    print(f"📄 檢索到 {len(documents)} 個文件")
    return documents

# 為了向後兼容，保留 VectorStore 別名
VectorStore = ChromaVectorStore
