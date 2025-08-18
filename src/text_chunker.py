import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TextChunk:
    """文本塊結構"""
    content: str
    metadata: Dict[str, Any] = None
    chunk_id: str = None
    start_index: int = 0
    end_index: int = 0

class TextChunker:
    """文本分塊器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        min_chunk_size: int = 100
    ):
        """
        初始化文本分塊器
        
        Args:
            chunk_size: 每個塊的最大字符數
            chunk_overlap: 相鄰塊之間的重疊字符數
            separator: 用於分割的自然分隔符
            min_chunk_size: 最小塊大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.min_chunk_size = min_chunk_size
    
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
        
        # 首先按自然分隔符分割
        segments = self._split_by_separator(text)
        
        chunks = []
        current_chunk = ""
        start_index = 0
        
        for segment in segments:
            # 如果當前塊加上新段落會超過限制
            if len(current_chunk) + len(segment) > self.chunk_size and current_chunk:
                # 保存當前塊
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        metadata=metadata.copy() if metadata else {},
                        start_index=start_index,
                        end_index=start_index + len(current_chunk)
                    )
                    chunks.append(chunk)
                
                # 開始新塊，包含重疊部分
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + segment
                start_index = start_index + overlap_start
            else:
                current_chunk += segment
            
            # 如果當前塊已經達到最小大小且超過限制，強制分割
            if len(current_chunk) >= self.chunk_size:
                # 嘗試在句子邊界分割
                split_point = self._find_sentence_boundary(current_chunk, self.chunk_size)
                if split_point > self.min_chunk_size:
                    chunk = TextChunk(
                        content=current_chunk[:split_point].strip(),
                        metadata=metadata.copy() if metadata else {},
                        start_index=start_index,
                        end_index=start_index + split_point
                    )
                    chunks.append(chunk)
                    
                    # 開始新塊，包含重疊
                    overlap_start = max(0, split_point - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    start_index = start_index + overlap_start
        
        # 添加最後一個塊
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk = TextChunk(
                content=current_chunk.strip(),
                metadata=metadata.copy() if metadata else {},
                start_index=start_index,
                end_index=start_index + len(current_chunk)
            )
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
        
        # 為每個塊生成唯一 ID
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"chunk_{i}_{hash(chunk.content) % 10000}"
            if chunk.metadata:
                chunk.metadata.update({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.content)
                })
        
        return chunks
    
    def _split_by_separator(self, text: str) -> List[str]:
        """按分隔符分割文本"""
        if not self.separator:
            return [text]
        
        segments = text.split(self.separator)
        # 重新添加分隔符（除了最後一個）
        result = []
        for i, segment in enumerate(segments):
            if i < len(segments) - 1:
                result.append(segment + self.separator)
            else:
                result.append(segment)
        return result
    
    def _find_sentence_boundary(self, text: str, max_length: int) -> int:
        """在句子邊界尋找分割點"""
        if len(text) <= max_length:
            return len(text)
        
        # 尋找句子結束標記
        sentence_endings = ['.', '!', '?', '。', '！', '？', '\n']
        
        for i in range(max_length, max(0, max_length - 100), -1):
            if text[i] in sentence_endings:
                return i + 1
        
        # 如果找不到句子邊界，尋找其他分隔符
        other_separators = [',', '，', ';', '；', ':', '：', ' ']
        for i in range(max_length, max(0, max_length - 50), -1):
            if text[i] in other_separators:
                return i + 1
        
        # 如果都找不到，強制分割
        return max_length

class DocumentChunker:
    """文檔分塊器，用於處理 Document 對象"""
    
    def __init__(self, chunker: TextChunker = None):
        """
        初始化文檔分塊器
        
        Args:
            chunker: 文本分塊器實例
        """
        self.chunker = chunker or TextChunker()
    
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
