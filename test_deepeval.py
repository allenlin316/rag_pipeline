import os
import asyncio
import json
from typing import Optional
from litellm import completion, acompletion
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
import re

# 載入環境變數
load_dotenv()


class CustomLLMJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "openai/gemma-3-27b-it",
        base_url: str = "https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai/",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        # 先初始化屬性，再調用父類的 __init__
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("GENERATOR_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model_loaded = False
        
        # 設定 LiteLLM 環境變數
        os.environ["LITELLM_BASE_URL"] = self.base_url
        if self.api_key:
            os.environ["LITELLM_API_KEY"] = self.api_key
        
        # 最後調用父類的 __init__
        super().__init__()

    def get_model_name(self):
        return f"Custom-{self.model}"

    def load_model(self):
        # 對於 API 調用，我們不需要實際載入模型
        # 只需要標記為已載入並返回 None 或字串
        self._model_loaded = True
        return self.model  # 返回模型名稱字串而不是 self

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        
        # 創建 JSON 格式的提示，類似於 instructor 的 GEMINI_JSON 模式
        json_prompt = self._create_json_prompt(prompt, schema)
        
        # 調用 LiteLLM API
        response = completion(
            model=client,
            messages=[
                {"role": "system", "content": "你是一個專業的 AI 助手，專門生成符合指定 JSON schema 的回應。請確保所有回應都是有效的 JSON 格式。"},
                {"role": "user", "content": json_prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # 提取並解析回應
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            json_data = self._extract_json_from_response(content)
            
            # 驗證並返回符合 schema 的對象
            try:
                return schema(**json_data)
            except Exception as e:
                print(f"Schema 驗證失敗: {e}")
                return schema()
        else:
            return schema()

    def _create_json_prompt(self, prompt: str, schema: BaseModel) -> str:
        """創建包含 JSON schema 的提示，類似於 instructor 的 GEMINI_JSON 模式"""
        schema_json = schema.model_json_schema()
        
        # 創建類似 instructor GEMINI_JSON 模式的提示
        json_prompt = f"""
請根據以下 JSON schema 回答問題。你的回答必須是有效的 JSON 格式，完全符合提供的 schema。

JSON Schema:
{json.dumps(schema_json, indent=2, ensure_ascii=False)}

問題: {prompt}

請確保你的回答是有效的 JSON 格式，不要包含任何額外的文字說明。
"""
        return json_prompt

    def _extract_json_from_response(self, response_text: str) -> dict:
        """從回應中提取 JSON"""
        try:
            # 首先嘗試直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果失敗，嘗試提取 JSON 部分
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            
            # 如果都失敗，返回錯誤信息
            return {"error": "無法解析 JSON", "raw_response": response_text}

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)


# 測試代碼
if __name__ == "__main__":
    # 定義一個測試用的 schema
    class TestResponse(BaseModel):
        answer: str
        confidence: float
        reasoning: str
    
    custom_llm = CustomLLMJudge()
    
    # 測試基本功能
    print("=== 測試基本功能 ===")
    try:
        result = custom_llm.generate("用繁體中文告訴我你叫什麼名字", TestResponse)
        print(f"回答: {result.answer}")
        print(f"信心度: {result.confidence}")
        print(f"推理: {result.reasoning}")
    except Exception as e:
        print(f"基本功能測試失敗: {e}")
    
    # 測試 DeepEval 評估
    print("\n=== 測試 DeepEval 評估 ===")
    try:
        # metric = AnswerRelevancyMetric(model=custom_llm)
        metric = FaithfulnessMetric(model=custom_llm)
        #contextual_precision = ContextualPrecisionMetric(model=custom_llm)
        #contextual_recall = ContextualRecallMetric()
        #contextual_relevancy = ContextualRelevancyMetric()
        test_case = LLMTestCase(
            input="我是F-1簽證持有者，畢業後可以在美國停留多久？",
            actual_output="你可以在完成學位後停留30天。",
            expected_output="你可以在完成學位後停留60天。",
            retrieval_context=[
                    """如果你在美國持有F-1簽證，你可以在完成學位後停留60天，
                    除非你已經申請並獲准參加OPT（可選實習訓練）。"""
            ]
        )
        #contextual_precision.measure(test_case)
        #print("Score: ", contextual_precision.score)
        #print("Reason: ", contextual_precision.reason)
        metric.measure(test_case)
        print(f"分數: {metric.score}")
        print(f"原因: {metric.reason}")
    except Exception as e:
        print(f"DeepEval 評估失敗: {e}")
