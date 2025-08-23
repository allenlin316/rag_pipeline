import re
from typing import List, Dict, Any
from dataclasses import dataclass
from .text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, Language

@dataclass
class TextChunk:
    """文本塊結構"""
    content: str
    metadata: Dict[str, Any] = None
    chunk_id: str = None
    start_index: int = 0
    end_index: int = 0

class TextChunker:
    """文本分塊器 - 使用新的 langchain 風格分割器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        min_chunk_size: int = 100,
        use_recursive: bool = True,
        language: Language = None
    ):
        """
        初始化文本分塊器
        
        Args:
            chunk_size: 每個塊的最大字符數
            chunk_overlap: 相鄰塊之間的重疊字符數
            separator: 用於分割的自然分隔符
            min_chunk_size: 最小塊大小
            use_recursive: 是否使用遞歸分割器
            language: 程式語言類型（用於語言特定的分割）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.min_chunk_size = min_chunk_size
        self.use_recursive = use_recursive
        self.language = language
        
        # 初始化分割器
        if use_recursive:
            if language:
                self.splitter = RecursiveCharacterTextSplitter.from_language(
                    language,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
        else:
            self.splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        將文本分割成塊
        
        Args:
            text: 要分割的文本
            metadata: 原始文檔的元數據
        
        Returns:
            文本塊列表
        """
        if not text.strip():
            return []
        
        # 使用新的分割器進行分割
        text_chunks = self.splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                # 計算在原文中的位置（近似）
                start_index = text.find(chunk_text)
                if start_index == -1:
                    start_index = i * self.chunk_size  # 近似位置
                
                chunk = TextChunk(
                    content=chunk_text.strip(),
                    metadata=metadata.copy() if metadata else {},
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )
                
                # 添加分割器相關的元數據
                if chunk.metadata:
                    chunk.metadata.update({
                        "splitter_type": "recursive" if self.use_recursive else "character",
                        "language": self.language.value if self.language else None,
                        "chunk_id": f"chunk_{i}_{hash(chunk_text) % 10000}",
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "chunk_size": len(chunk_text)
                    })
                
                chunks.append(chunk)
        
        # Fallback: 若沒有產生任何 chunk，就把整篇原文作為單一 chunk
        if not chunks:
            whole = text.strip()
            if whole:
                chunks = [TextChunk(
                    content=whole,
                    metadata=metadata.copy() if metadata else {},
                    start_index=0,
                    end_index=len(whole)
                )]
                if chunks[0].metadata:
                    chunks[0].metadata.update({
                        "splitter_type": "fallback",
                        "chunk_id": f"chunk_0_{hash(whole) % 10000}",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(whole)
                    })
        
        return chunks
    


class DocumentChunker:
    """文檔分塊器，用於處理 Document 對象"""
    
    def __init__(self, chunker: TextChunker = None):
        """
        初始化文檔分塊器
        
        Args:
            chunker: 文本分塊器實例
        """
        self.chunker = chunker or TextChunker(
            use_recursive=True,
            language=None
        )
    
    def chunk_documents(self, documents: List['Document']) -> List['Document']:
        """
        將文檔列表分割成塊
        
        Args:
            documents: 原始文檔列表
        
        Returns:
            分塊後的文檔列表
        """
        chunked_documents = []

        # 動態獲取 Document 類，避免循環導入
        def _get_document_class():
            from .retriever import Document  # 延遲導入
            return Document
        Document = _get_document_class()

        for doc in documents:
            # 分割文本
            text_chunks = self.chunker.split_text(doc.content, doc.metadata)
            
            # 轉換為 Document 對象
            for chunk in text_chunks:
                chunked_doc = Document(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=doc.score
                )
                chunked_documents.append(chunked_doc)
        
        return chunked_documents

# 為了避免循環導入，使用字符串類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .retriever import Document
