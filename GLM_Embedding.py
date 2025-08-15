import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv

# 加载 .env 文件中的环境变量（推荐的安全做法）
_ = load_dotenv(find_dotenv())

def zhipu_embedding(text: str):
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("❌ ZHIPUAI_API_KEY 没有在 .env 文件或环境变量中设置！")

    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
    return response

text = '要生成 embedding 的输入文本，字符串形式。'
response = zhipu_embedding(text=text)

print(f'response类型为：{type(response)}')
print(f'embedding类型为：{response.object}')
print(f'生成embedding的model为：{response.model}')
print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:10]}')
