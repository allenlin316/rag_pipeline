import requests
from typing import List
from config import get_config

# 定義 Document 類型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .retriever import Document

class GeneratorAPI:
    """Generator API 客戶端"""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, max_tokens: int = None, temperature: float = None, skip_special_tokens: bool = None):
        config = get_config()
        self.api_key = config.generator_api_key
        self.base_url = base_url or config.generator_base_url
        self.model = model or config.generator_model
        self.max_tokens = max_tokens or config.max_tokens
        self.temperature = temperature or config.temperature
        # 處理 skip_special_tokens 邏輯
        # 如果沒有給 --skip-special-tokens 參數，則為 True
        # 如果有給 --skip-special-tokens 參數，則為 False
        config_skip_special_tokens = getattr(config, 'skip_special_tokens', True)
        self.skip_special_tokens = skip_special_tokens if skip_special_tokens is not None else (not config_skip_special_tokens)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate(self, prompt: str, model: str = None, max_tokens: int = None, temperature: float = None, skip_special_tokens: bool = None) -> str:
        """生成回答"""
        data = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": "你是一個專業的AI助手，請用繁體中文回答用戶的問題。回答要準確、有幫助且易於理解。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        
        # 處理 skip_special_tokens 參數
        # 優先級：函數參數 > 配置
        should_skip_special_tokens = skip_special_tokens
        if should_skip_special_tokens is None:
            should_skip_special_tokens = self.skip_special_tokens
        
        # 如果設定了 skip_special_tokens，添加到 extra_body 中
        if should_skip_special_tokens is not None:
            data["extra_body"] = {
                "skip_special_tokens": should_skip_special_tokens
            }
            print(f"🔧 使用 skip_special_tokens={should_skip_special_tokens} (在 extra_body 中)")
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Generator API 錯誤: {response.status_code} - {response.text}")

def generator(query: str, documents: List["Document"]) -> str:
    """
    Generator 函數：基於檢索到的文件生成回答
    
    Args:
        query: 查詢文本
        documents: 相關文件列表
    
    Returns:
        生成的回答
    """
    generator_api = GeneratorAPI()
    
    print(f"🤖 使用 {generator_api.model} 生成回答...")
    
    if not documents:
        return "抱歉，我沒有找到相關的資訊來回答您的問題。"
    
    # 構建 prompt
    context = "\n\n".join([f"文件 {i+1}: {doc.content}" for i, doc in enumerate(documents)])
    
    prompt = f"""基於以下相關文件，請用繁體中文回答用戶的問題。如果文件中沒有相關資訊，請誠實地說不知道。

相關文件：
{context}

用戶問題：{query}

請用繁體中文提供準確、有幫助的回答："""

    
    
    try:
        answer = generator_api.generate(prompt)
        print(f"✅ 回答生成完成")
        return answer
    except Exception as e:
        print(f"⚠️ Generator API 錯誤: {e}")
        return f"抱歉，生成回答時發生錯誤: {e}"
