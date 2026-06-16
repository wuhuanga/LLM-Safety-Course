"""
LLM客户端 - 支持千问和OpenAI
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class BaseLLMClient:
    """LLM客户端基类"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话接口"""
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        """嵌入接口"""
        raise NotImplementedError


class QwenClient(BaseLLMClient):
    """千问客户端"""

    def __init__(self, model: str = "qwen-max", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        try:
            import dashscope
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """使用千问进行对话"""
        from dashscope import Generation

        # 合并配置
        config = {
            "model": self.model,
            "messages": messages,
            "result_format": "message",
            **self.config,
            **kwargs
        }

        # 调用API
        response = Generation.call(**config)

        if response.status_code != 200:
            raise Exception(f"千问API错误: {response.message}")

        return response.output.choices[0].message.content

    def embed(self, text: str) -> List[float]:
        """使用千问生成文本嵌入"""
        from dashscope import TextEmbedding

        response = TextEmbedding.call(
            model="text-embedding-v2",
            input=text,
            api_key=self.api_key
        )

        if response.status_code != 200:
            raise Exception(f"千问嵌入API错误: {response.message}")

        return response.output["embeddings"][0]["embedding"]


class OpenAIClient(BaseLLMClient):
    """OpenAI兼容客户端（用于嵌入或其他用途）"""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """使用OpenAI进行对话"""
        config = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs
        }

        response = self.client.chat.completions.create(**config)
        return response.choices[0].message.content

    def embed(self, text: str) -> List[float]:
        """使用OpenAI生成文本嵌入"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


class SentenceTransformerClient(BaseLLMClient):
    """本地嵌入客户端（免费但需要下载模型）"""

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu", **kwargs):
        super().__init__(model, **kwargs)
        self.device = device

        try:
            from sentence_transformers import SentenceTransformer
            print(f"正在加载嵌入模型: {model}")
            self.model = SentenceTransformer(model, device=device)
            print(f"模型加载完成")
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """此客户端不支持对话"""
        raise NotImplementedError("SentenceTransformer只支持嵌入，不支持对话")

    def embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        return self.model.encode(text, convert_to_numpy=True).tolist()


def create_llm_client(provider: str = "dashscope", model: str = "qwen-max", **kwargs) -> BaseLLMClient:
    """创建LLM客户端工厂函数

    Args:
        provider: 提供商，可选: dashscope, openai, sentence-transformers
        model: 模型名称
        **kwargs: 其他配置参数

    Returns:
        LLM客户端实例
    """
    providers = {
        "dashscope": QwenClient,
        "qwen": QwenClient,
        "openai": OpenAIClient,
        "sentence-transformers": SentenceTransformerClient,
    }

    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(f"不支持的provider: {provider}。可选: {list(providers.keys())}")

    return providers[provider_lower](model=model, **kwargs)


# 快捷函数
def get_llm_client(**kwargs) -> BaseLLMClient:
    """从环境变量或配置创建LLM客户端"""
    provider = kwargs.pop("provider", None) or os.getenv("LLM_PROVIDER", "dashscope")
    model = kwargs.pop("model", None) or os.getenv("LLM_MODEL", "qwen-max")
    return create_llm_client(provider=provider, model=model, **kwargs)


def get_embedding_client(**kwargs) -> BaseLLMClient:
    """从环境变量或配置创建嵌入客户端"""
    provider = kwargs.pop("provider", None) or os.getenv("EMBEDDING_PROVIDER", "dashscope")
    model = kwargs.pop("model", None) or os.getenv("EMBEDDING_MODEL", "qwen-text-embedding-v2")
    return create_llm_client(provider=provider, model=model, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 测试千问
    print("测试千问API...")
    try:
        client = get_llm_client(model="qwen-turbo")
        messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
        response = client.chat(messages)
        print(f"千问回复: {response}")
    except Exception as e:
        print(f"千问测试失败: {e}")

    print("\n测试千问嵌入...")
    try:
        embed_client = get_embedding_client()
        embedding = embed_client.embed("这是一个测试文本")
        print(f"嵌入维度: {len(embedding)}")
    except Exception as e:
        print(f"嵌入测试失败: {e}")