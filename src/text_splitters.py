from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union
from abc import ABC, abstractmethod


class Language:
    """支援的程式語言枚舉"""
    C = "c"
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    ELIXIR = "elixir"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    CSHARP = "csharp"
    SOL = "sol"
    COBOL = "cobol"
    LUA = "lua"
    HASKELL = "haskell"
    POWERSHELL = "powershell"
    VISUALBASIC6 = "visualbasic6"


class TextSplitter(ABC):
    """文本分割器基類"""
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Optional[callable] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = False,
        **kwargs: Any,
    ) -> None:
        """初始化文本分割器"""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function or len
        self._keep_separator = keep_separator
    
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """分割文本"""
        pass
    
    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """合併分割後的文本塊"""
        if not splits:
            return []
        
        merged = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self._length_function(split)
            
            # 如果當前塊加上新分割會超過大小限制
            if current_length + split_length > self._chunk_size and current_chunk:
                # 保存當前塊
                merged.append(separator.join(current_chunk))
                
                # 開始新塊，保留重疊部分
                overlap_start = max(0, len(current_chunk) - self._chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(self._length_function(s) for s in current_chunk)
            
            current_chunk.append(split)
            current_length += split_length
        
        # 添加最後一個塊
        if current_chunk:
            merged.append(separator.join(current_chunk))
        
        return merged


class CharacterTextSplitter(TextSplitter):
    """基於字符的文本分割器"""

    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """創建新的文本分割器"""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> list[str]:
        """分割文本而不重新插入分隔符"""
        # 1. 確定分割模式：原始正則表達式或轉義字面量
        sep_pattern = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )

        # 2. 初始分割（如果請求則保留分隔符）
        splits = _split_text_with_regex(
            text, sep_pattern, keep_separator=self._keep_separator
        )

        # 3. 檢測零寬度前瞻後顧，這樣我們永遠不會重新插入它
        lookaround_prefixes = ("(?=", "(?<!", "(?<=", "(?!")
        is_lookaround = self._is_separator_regex and any(
            self._separator.startswith(p) for p in lookaround_prefixes
        )

        # 4. 決定合併分隔符：
        #    - 如果 keep_separator 或 lookaround -> 不重新插入
        #    - 否則 -> 重新插入字面量分隔符
        merge_sep = ""
        if not (self._keep_separator or is_lookaround):
            merge_sep = self._separator

        # 5. 合併相鄰分割並返回
        return self._merge_splits(splits, merge_sep)


def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: Union[bool, Literal["start", "end"]]
) -> list[str]:
    """使用正則表達式分割文本"""
    # 現在我們有了分隔符，分割文本
    if separator:
        if keep_separator:
            # 模式中的括號將分隔符保留在結果中
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                ([*splits, _splits[-1]])
                if keep_separator == "end"
                else ([_splits[0], *splits])
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """遞歸字符文本分割器
    
    遞歸地嘗試使用不同的字符進行分割，找到一個有效的方法。
    """

    def __init__(
        self,
        separators: Optional[list[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """創建新的文本分割器"""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """分割傳入的文本並返回塊"""
        final_chunks = []
        # 獲取適當的分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, _separator, keep_separator=self._keep_separator
        )

        # 現在開始合併，遞歸地分割較長的文本
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> list[str]:
        """根據預定義的分隔符將輸入文本分割成較小的塊"""
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
        cls, language: Language, **kwargs: Any
    ) -> RecursiveCharacterTextSplitter:
        """根據特定語言返回此類的實例"""
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> list[str]:
        """獲取給定語言的特定分隔符列表"""
        if language in (Language.C, Language.CPP):
            return [
                # 沿類定義分割
                "\nclass ",
                # 沿函數定義分割
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.GO:
            return [
                # 沿函數定義分割
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.JAVA:
            return [
                # 沿類定義分割
                "\nclass ",
                # 沿方法定義分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.KOTLIN:
            return [
                # 沿類定義分割
                "\nclass ",
                # 沿方法定義分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.JS:
            return [
                # 沿函數定義分割
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # 沿類定義分割
                "\nclass ",
                # 沿函數定義分割
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PHP:
            return [
                # 沿函數定義分割
                "\nfunction ",
                # 沿類定義分割
                "\nclass ",
                # 沿控制流語句分割
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PROTO:
            return [
                # 沿消息定義分割
                "\nmessage ",
                # 沿服務定義分割
                "\nservice ",
                # 沿枚舉定義分割
                "\nenum ",
                # 沿選項定義分割
                "\noption ",
                # 沿導入語句分割
                "\nimport ",
                # 沿語法聲明分割
                "\nsyntax ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.PYTHON:
            return [
                # 首先，嘗試沿類定義分割
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # 現在按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RST:
            return [
                # 沿章節標題分割
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # 沿指令標記分割
                "\n\n.. *\n\n",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RUBY:
            return [
                # 沿方法定義分割
                "\ndef ",
                "\nclass ",
                # 沿控制流語句分割
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.ELIXIR:
            return [
                # 沿方法函數和模組定義分割
                "\ndef ",
                "\ndefp ",
                "\ndefmodule ",
                "\ndefprotocol ",
                "\ndefmacro ",
                "\ndefmacrop ",
                # 沿控制流語句分割
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\ncase ",
                "\ncond ",
                "\nwith ",
                "\nfor ",
                "\ndo ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.RUST:
            return [
                # 沿函數定義分割
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # 沿控制流語句分割
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SCALA:
            return [
                # 沿類定義分割
                "\nclass ",
                "\nobject ",
                # 沿方法定義分割
                "\ndef ",
                "\nval ",
                "\nvar ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SWIFT:
            return [
                # 沿函數定義分割
                "\nfunc ",
                # 沿類定義分割
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.MARKDOWN:
            return [
                # 首先，嘗試沿 Markdown 標題分割（從級別 2 開始）
                "\n#{1,6} ",
                # 注意：這裡不處理標題的替代語法（如下）
                # 標題級別 2
                # ---------------
                # 代碼塊結束
                "```\n",
                # 水平線
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # 注意：此分割器不處理由 ***、--- 或 ___ 的三個或更多定義的水平線
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.LATEX:
            return [
                # 首先，嘗試沿 Latex 章節分割
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # 現在按環境分割
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # 現在按數學環境分割
                "\n\\\\begin{align}",
                "$$",
                "$",
                # 現在按正常行類型分割
                " ",
                "",
            ]
        if language == Language.HTML:
            return [
                # 首先，嘗試沿 HTML 標籤分割
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        if language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # 沿類定義分割
                "\nclass ",
                "\nabstract ",
                # 沿方法定義分割
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # 沿控制流語句分割
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # 按異常分割
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.SOL:
            return [
                # 沿編譯器信息定義分割
                "\npragma ",
                "\nusing ",
                # 沿合約定義分割
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # 沿方法定義分割
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.COBOL:
            return [
                # 沿分區分割
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # 沿 DATA DIVISION 內的節分割
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # 沿 PROCEDURE DIVISION 內的節分割
                "\nINPUT-OUTPUT SECTION.",
                # 沿段落和常見語句分割
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # 按正常行類型分割
                "\n",
                " ",
                "",
            ]
        if language == Language.LUA:
            return [
                # 沿變量和表定義分割
                "\nlocal ",
                # 沿函數定義分割
                "\nfunction ",
                # 沿控制流語句分割
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.HASKELL:
            return [
                # 沿函數定義分割
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                # 沿類型聲明分割
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                "\n:: ",
                # 沿模組聲明分割
                "\nmodule ",
                # 沿導入語句分割
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                # 沿類型類聲明分割
                "\nclass ",
                "\ninstance ",
                # 沿 case 表達式分割
                "\ncase ",
                # 沿函數定義中的守衛分割
                "\n| ",
                # 沿記錄字段聲明分割
                "\ndata ",
                "\n= {",
                "\n, ",
                # 按正常行類型分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.POWERSHELL:
            return [
                # 沿函數定義分割
                "\nfunction ",
                # 沿參數聲明分割（轉義括號）
                "\nparam ",
                # 沿控制流語句分割
                "\nif ",
                "\nforeach ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                # 沿類定義分割（適用於 PowerShell 5.0 及以上版本）
                "\nclass ",
                # 沿 try-catch-finally 塊分割
                "\ntry ",
                "\ncatch ",
                "\nfinally ",
                # 按正常行和空空格分割
                "\n\n",
                "\n",
                " ",
                "",
            ]
        if language == Language.VISUALBASIC6:
            vis = r"(?:Public|Private|Friend|Global|Static)\s+"
            return [
                # 沿定義分割
                rf"\n(?!End\s){vis}?Sub\s+",
                rf"\n(?!End\s){vis}?Function\s+",
                rf"\n(?!End\s){vis}?Property\s+(?:Get|Let|Set)\s+",
                rf"\n(?!End\s){vis}?Type\s+",
                rf"\n(?!End\s){vis}?Enum\s+",
                # 沿控制流語句分割
                r"\n(?!End\s)If\s+",
                r"\nElseIf\s+",
                r"\nElse\s+",
                r"\nSelect\s+Case\s+",
                r"\nCase\s+",
                r"\nFor\s+",
                r"\nDo\s+",
                r"\nWhile\s+",
                r"\nWith\s+",
                # 按正常行類型分割
                r"\n\n",
                r"\n",
                " ",
                "",
            ]

        if language in Language._value2member_map_:
            msg = f"Language {language} is not implemented yet!"
            raise ValueError(msg)
        msg = (
            f"Language {language} is not supported! Please choose from {list(Language)}"
        )
        raise ValueError(msg)
