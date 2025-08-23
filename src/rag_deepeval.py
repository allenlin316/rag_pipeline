import os
import asyncio
from typing import Optional, Literal, List
from openai import OpenAI, AsyncOpenAI
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
from pydantic import BaseModel, Field

# 載入環境變數
load_dotenv()

class CustomLLMJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "gpt-oss-120b",
        base_url: str = "https://litellm-ekkks8gsocw.dgx-coolify.apmic.ai",
        api_key: Optional[str] = None,
        max_tokens: int = 10000,
        temperature: float = 0.7
    ):
        # 先初始化屬性，再調用父類的 __init__
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("GENERATOR_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model_loaded = False
        
        # 初始化 OpenAI client
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=f"{self.base_url}/v1"
        )
        
        # 初始化異步 OpenAI client
        self.async_client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=f"{self.base_url}/v1"
        )
        
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
        
        # 創建系統提示
        system_prompt = "你是一個專業的 AI 助手，專門生成符合指定結構的回應。請確保所有回應都完全符合提供的 schema 格式。"
        
        
        try:
            # 使用 OpenAI client 的結構化解析功能
            parsed = self.client.beta.chat.completions.parse(
                model=client,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 取得結構化結果
            result = parsed.choices[0].message.parsed
            return result
            
        except Exception as e:
            print(f"結構化解析失敗: {e}")
            print(f"原始提示長度: {len(prompt)} 字符")
            print(f"使用的 max_tokens: {prompt}")
            # 返回一個默認的 schema 實例
            return self._create_default_schema_instance(schema)

    def _create_default_schema_instance(self, schema: BaseModel) -> BaseModel:
        """創建 schema 的默認實例"""
        try:
            # 嘗試創建一個默認實例
            return schema()
        except Exception:
            # 如果無法創建默認實例，嘗試手動創建
            schema_fields = schema.model_fields
            default_values = {}
            
            for field_name, field_info in schema_fields.items():
                if hasattr(field_info, 'default') and field_info.default is not None:
                    default_values[field_name] = field_info.default
                elif field_info.annotation == str:
                    default_values[field_name] = "無法解析"
                elif field_info.annotation == Optional[str]:
                    default_values[field_name] = None
                elif hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ == list:
                    # 處理 List 類型
                    default_values[field_name] = []
                elif field_info.annotation == float:
                    default_values[field_name] = 0.0
                elif field_info.annotation == int:
                    default_values[field_name] = 0
                elif field_info.annotation == bool:
                    default_values[field_name] = False
                else:
                    default_values[field_name] = None
            
            return schema(**default_values)



    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        """異步版本的 generate 方法"""
        # 創建系統提示
        system_prompt = "你是一個專業的 AI 助手，專門生成符合指定結構的回應。請確保所有回應都完全符合提供的 schema 格式。"
        
        try:
            # 使用異步 OpenAI client 的結構化解析功能
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 取得結構化結果
            result = response.choices[0].message.parsed
            return result
            
        except Exception as e:
            print(f"異步結構化解析失敗: {e}")
            print(f"原始提示長度: {len(prompt)} 字符")
            print(f"使用的 max_tokens: {self.max_tokens}")
            # 返回一個默認的 schema 實例
            return self._create_default_schema_instance(schema)

def evaluate_rag_pipeline(
    query: str,
    actual_output: str,
    expected_output: str,
    retrieval_context: list,
    metrics: list = None,
    llm_judge_model: str = None
) -> dict:
    """
    評估 RAG Pipeline 的表現
    
    Args:
        query: 查詢文本
        actual_output: 實際輸出
        expected_output: 期望輸出
        retrieval_context: 檢索到的上下文
        metrics: 要評估的指標列表
    
    Returns:
        評估結果字典
    """
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy"]
    
    # 使用配置中的 llm-judge-model，如果沒有提供則使用默認值
    if llm_judge_model is None:
        try:
            from config import get_config
            config = get_config()
            llm_judge_model = getattr(config, 'llm_judge_model', 'gpt-oss-120b')
        except ImportError:
            llm_judge_model = 'gpt-oss-120b'
    
    custom_llm = CustomLLMJudge(model=llm_judge_model)
    results = {}
    
    # 創建測試案例
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    
    # 執行評估
    for metric_name in metrics:
        try:
            if metric_name == "faithfulness":
                metric = FaithfulnessMetric(model=custom_llm)
            elif metric_name == "answer_relevancy":
                metric = AnswerRelevancyMetric(model=custom_llm)
            elif metric_name == "contextual_precision":
                metric = ContextualPrecisionMetric(model=custom_llm)
            elif metric_name == "contextual_recall":
                metric = ContextualRecallMetric(model=custom_llm)
            elif metric_name == "contextual_relevancy":
                metric = ContextualRelevancyMetric(model=custom_llm)
            else:
                print(f"⚠️ 未知的評估指標: {metric_name}")
                continue
            
            metric.measure(test_case)
            results[metric_name] = {
                "score": metric.score,
                "reason": metric.reason
            }
            print(f"✅ {metric_name}: {metric.score}")
            
        except Exception as e:
            print(f"❌ {metric_name} 評估失敗: {e}")
            results[metric_name] = {
                "score": None,
                "reason": f"評估失敗: {e}"
            }
    
    return results
