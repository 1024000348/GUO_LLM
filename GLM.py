# from zhipuai_llm import ZhipuaiLLM
# from dotenv import find_dotenv, load_dotenv
# import os

# # 读取本地/项目的环境变量。

# # find_dotenv()寻找并定位.env文件的路径
# # load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# # 如果你设置的是全局的环境变量，这行代码则没有任何作用。
# _ = load_dotenv(find_dotenv())

# # 获取环境变量 API_KEY
# api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

# llm = ZhipuaiLLM(model_name="glm-4-flash", temperature=0.1, api_key=api_key)

# question  = "介绍一下你自己"
# response = zhipuai_model.invoke(question)
# print(response.content)




# GPT版本
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatZhipuAI  # ✅ 使用LangChain社区封装的智谱GLM接口

# 1️⃣ 加载 .env 文件
_ = load_dotenv(find_dotenv()) 

# 从环境变量读取 ZHIPUAI_API_KEY
ZHIPUAI_API_KEY = os.environ["ZHIPUAI_API_KEY"]

# 初始化 GLM 模型（glm-4.5-flash）
llm = ChatZhipuAI(model="glm-4.5-flash", temperature=0.1, api_key=ZHIPUAI_API_KEY)

# 调用模型
# print(llm.invoke("请你自我介绍一下自己！").content)
