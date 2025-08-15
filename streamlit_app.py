# 导入必要的 Python 库。
# --- 解决 chromadb 在云环境 sqlite 不兼容问题 ---
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
# ------------------------------------------------

import streamlit as st
from langchain_openai import ChatOpenAI
import os
import sys
from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# ===== 新增美化功能 START =====
# 设置网页标题、图标、布局
st.set_page_config(
    page_title="  ٩̋(๑˃́ꇴ˂̀๑)",
    page_icon="🤖",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
        /* 修改整体背景颜色 */
        .stApp {
            background-color: #f5f7fa;
        }
        /* 聊天气泡美化 */
        .stChatMessage {
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
        }
        /* 用户消息背景 */
        .stChatMessage.human {
            background-color: #DCF8C6;
        }
        /* AI消息背景 */
        .stChatMessage.ai {
            background-color: #E6E6FA;
        }
        /* 输入框样式 */
        .stChatInput input {
            background-color: #fff;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
        }
        /* 全局字体 */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
# ===== 新增美化功能 END =====


# 加载环境变量
_ = load_dotenv(find_dotenv())

# 将父目录放入系统路径中
sys.path.append("notebook/C3 搭建知识库") 

# 导入智谱 Embedding 和 LLM 封装
from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuaiLLM
from langchain_community.vectorstores import Chroma


# 定义get_retriever函数，该函数返回一个检索器
def get_retriever():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'D:/LLM/data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()


# 定义combine_docs函数， 该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


# 定义get_qa_history_chain函数，该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    api_key = os.environ["ZHIPUAI_API_KEY"]

    # 使用 glm-4-flash 模型
    llm = ZhipuaiLLM(model_name="glm-4-flash", temperature=0.1, api_key=api_key)

    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "你的名字叫做奶龙小朋友。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说哎呀～这个嘛…奶龙小朋友真的不太清楚啦～(>_<)。 "
        "请使用可爱、类似奶龙傻傻的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain


# 定义gen_response函数，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


# 定义main函数，该函数制定显示效果与逻辑
def main():
    st.set_page_config(
    page_title="  ٩̋(๑˃́ꇴ˂̀๑)",
    page_icon="🤖",
    layout="wide"  # 宽屏布局
    )
    st.markdown('###   ٩̋(๑˃́ꇴ˂̀๑)')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            avatar = "🧑" if message[0] == "human" else "🤖"
            with messages.chat_message(message[0], avatar=avatar):
                st.markdown(f"<div style='font-size:16px;'>{message[1]}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human", avatar="🧑"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai", avatar="🤖"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()


# 调用
# python -m streamlit run D:/LLM/model/GLM/streamlit_app.py
