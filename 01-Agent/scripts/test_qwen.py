"""测试千问API连接"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.client import get_llm_client

load_dotenv()

print("=" * 60)
print("千问API连接测试")
print("=" * 60)

try:
    client = get_llm_client()

    messages = [
        {"role": "user", "content": "请回答：1+1等于多少？"}
    ]

    print("\n发送测试请求...")
    response = client.chat(messages, max_tokens=100, temperature=0)

    print(f"\n响应: {response}")

    if "2" in response or "二" in response:
        print("\n[成功] API连接正常")
    else:
        print("\n[警告] 响应内容异常")

except Exception as e:
    print(f"\n[失败] {e}")
    print("\n请检查：")
    print("1. .env文件是否存在")
    print("2. DASHSCOPE_API_KEY是否正确配置")
    print("3. 网络连接是否正常")

print("\n" + "=" * 60)