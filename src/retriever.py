import requests
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
try:
    from tqdm import tqdm  # 可選依賴：若不存在則退回簡單列印
except Exception:
    tqdm = None
from dataclasses import dataclass
from config import get_config
from .text_chunker import DocumentChunker, TextChunker
from .text_splitters import Language

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
        self.api_key = config.generator_api_key
        self.base_url = base_url or config.retriever_base_url
        self.model = model or config.embedding_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = requests.Session()
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """獲取文本的 embedding"""
        if not text or not text.strip():
            raise Exception("文本內容為空")
        
        data = {
            "input": text,
            "model": model or self.model
        }
        
        try:
            r = self.session.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=data,
                timeout=30  # 添加超時
            )
            r.raise_for_status()
            result = r.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    raise Exception("API 返回的 embedding 為空")
            else:
                raise Exception("API 返回的數據格式不正確")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"網絡請求錯誤: {e}")
        except Exception as e:
            raise Exception(f"獲取 embedding 時發生錯誤: {e}")

    def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """獲取多個文本的 embedding"""
        if not texts:
            return []
        data = {
            "input": texts,
            "model": model or self.model
        }
        try:
            r = self.session.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=data,
                timeout=60  # 添加超時
            )
            r.raise_for_status()
            result = r.json()
            if "data" in result and len(result["data"]) > 0:
                return [d["embedding"] for d in result["data"]]
            else:
                raise Exception("API 返回的數據格式不正確")
        except requests.exceptions.RequestException as e:
            raise Exception(f"網絡請求錯誤: {e}")
        except Exception as e:
            raise Exception(f"獲取 embeddings 時發生錯誤: {e}")

class ChromaVectorStore:
    """使用 Chroma 的向量儲存"""
    def __init__(
        self, 
        collection_name: str = "rag_documents", 
        persist_directory: str = "./chroma_db",
        enable_chunking: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        use_recursive: bool = True,
        language: Language = None,
        reset: bool = False
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
        
        # 重置集合（可選）
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"🗑️ 已重置集合: {collection_name}")
            except Exception:
                pass

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
                    chunk_overlap=chunk_overlap,
                    use_recursive=use_recursive,
                    language=language
                )
            )
            chunker_type = "遞歸分割器" if use_recursive else "字符分割器"
            language_info = f" ({language.value})" if language else ""
            print(f"✅ 啟用文本分塊: {chunker_type}{language_info} (chunk_size={chunk_size}, overlap={chunk_overlap})")
        else:
            self.chunker = None
            print("⏭️ 禁用文本分塊")
        
        print(f"使用 {self.embedding_api.model} 進行 embedding")
        
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
        
        total_docs = len(documents_to_add)
        iterator = documents_to_add
        if tqdm is not None:
            iterator = tqdm(documents_to_add, desc="Embedding documents", unit="doc")

        for i, doc in enumerate(iterator):
            # 生成唯一 ID - 使用更可靠的方式
            import uuid
            doc_id = f"doc_{i}_{uuid.uuid4().hex[:8]}"
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

            # 簡易進度輸出（當未安裝 tqdm 時）
            if tqdm is None and total_docs > 0:
                if (i + 1) == 1 or (i + 1) % 5 == 0 or (i + 1) == total_docs:
                    print(f"⏳ 進度: {i + 1}/{total_docs}")
        
        # 若沒有可添加的文檔，直接返回
        if not ids and len(documents_to_add) == 0:
            print("⚠️ 沒有可添加的文件，跳過")
            return

        # 批量添加到 Chroma
        try:
            if not ids or not embeddings:
                print("⚠️ 空的 ids 或 embeddings，跳過添加")
                return
            
            # 使用批次 API 進行嵌入（修正版本）
            batch_size = 64 # 設定批次大小
            final_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                batch_embs = self.embedding_api.get_embeddings(batch_texts)
                
                if len(batch_embs) != len(batch_texts):
                    raise Exception(f"批次嵌入失敗，預期 {len(batch_texts)} 個 embedding，但只收到 {len(batch_embs)} 個")
                
                for j, emb in enumerate(batch_embs):
                    if emb and len(emb) > 0:
                        final_embeddings.append(emb)
                    else:
                        print(f"⚠️ 批次嵌入失敗 (文檔 {i+j+1}): 獲取到的 embedding 為空")
                        # 如果 API 失敗，使用零向量作為備用
                        final_embeddings.append([0.0] * embedding_dimension)

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=final_embeddings
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
            
            # 轉換為 Document 物件並按分數排序
            all_documents = []
            
            if results["documents"] and results["documents"][0]:
                print(f"🔍 檢索到 {len(results['documents'][0])} 個文檔")
                print(f"🔍 距離值: {results['distances'][0]}")
                
                # 先處理所有文檔，計算分數
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # 將距離轉換為相似度分數 (Chroma 使用歐幾里得距離)
                    # 使用更好的轉換方法：1 - normalized_distance
                    if len(results["distances"][0]) > 1:
                        max_distance = max(results["distances"][0])
                        min_distance = min(results["distances"][0])
                        if max_distance > min_distance:
                            normalized_distance = (distance - min_distance) / (max_distance - min_distance)
                            similarity_score = 1.0 - normalized_distance
                        else:
                            similarity_score = 1.0  # 如果所有距離都相同
                    else:
                        # 如果只有一個結果，使用更合理的轉換方法
                        similarity_score = max(0.0, 1.0 - (distance / 10.0))  # 假設最大距離為10
                    
                    doc = Document(
                        content=doc_text,
                        metadata=metadata,
                        score=similarity_score
                    )
                    all_documents.append(doc)
                    print(f"🔍 文檔 {i+1} 分數: {similarity_score:.6f} (距離: {distance:.6f})")
                
                # 按分數排序（降序）
                all_documents.sort(key=lambda x: x.score, reverse=True)
                print(f"📊 按分數排序完成，最高分: {all_documents[0].score:.6f}, 最低分: {all_documents[-1].score:.6f}")
                
                # 去重並保留最高分的文檔
                unique_documents = []
                seen_contents = set()
                
                for doc in all_documents:
                    # 檢查是否為完全重複
                    if doc.content in seen_contents:
                        print(f"🔄 跳過重複內容 (分數: {doc.score:.6f})")
                        continue
                    
                    # 檢查是否為 overlap 重複（如果啟用了分塊且有 overlap）
                    is_overlap_duplicate = False
                    if self.enable_chunking and hasattr(self, 'chunker') and self.chunker:
                        for seen_content in seen_contents:
                            # 如果一個文本完全包含在另一個文本中，可能是 overlap 重複
                            if (len(doc.content) > len(seen_content) and seen_content in doc.content) or \
                               (len(seen_content) > len(doc.content) and doc.content in seen_content):
                                # 計算重疊比例
                                overlap_ratio = min(len(doc.content), len(seen_content)) / max(len(doc.content), len(seen_content))
                                if overlap_ratio > 0.8:  # 如果重疊超過80%，視為 overlap 重複
                                    print(f"🔄 跳過 overlap 重複 (分數: {doc.score:.6f}, 重疊比例: {overlap_ratio:.2f})")
                                    is_overlap_duplicate = True
                                    break
                    
                    if is_overlap_duplicate:
                        continue
                    
                    seen_contents.add(doc.content)
                    unique_documents.append(doc)
                    print(f"✅ 保留文檔 (分數: {doc.score:.6f})")
                
                # 返回前 top_k 個唯一文檔
                final_documents = unique_documents[:top_k]
                print(f"✅ 去重後返回 {len(final_documents)} 個唯一文檔 (最高分: {final_documents[0].score:.6f})")
                return final_documents
            
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
