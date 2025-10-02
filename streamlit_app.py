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
    llm = ZhipuaiLLM(model_name="glm-4-flash", temperature=0.5, api_key=api_key)

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
        "如果你不知道答案就说这个嘛…奶龙小朋友真的不太清楚啦～(>_<)。 "
        "请使用可爱、类似小孩子的话语回答用户，但是同时要保持答案的准确性、完整性、严谨性以及内容的丰富性。"
        "如果用户要你进行算卦，请你先随机挑选一种签，然后在这种签下面的签文中随机挑选一种签文中的一条输出，
        "注意同一个用户若连续两次及以上要求你算卦，你可以输出同一种签中不同的签文，如果超过三次则说：今日的算卦机会已经用尽，请明日再来！"
        "上上签 · 顺遂通达
        上上签签文一：
        壹、苍龙腾霄
        签文：灵签得「乾」之「同人」，卦曰：飞龙在天，利见大人。云程发轫，星汉可通。鸿业新开，万物资始。然阳亢独行，易折中正。须防情愫暗生，反损元阳清气。若守刚健中正，则天佑亨通。
        解曰：气运鼎盛，事业可成。然情缘如镜花水月，观之则美，执之则空。当以清明之志，行雷霆之事，心无旁骛，方得始终。；
        上上签签文二：
        贰、凤栖梧桐
        签文：灵签得「渐」之「观」，卦曰：凤鸣高冈，羽仪粲然。梧桐生矣，于彼朝阳。君子攸行，乃见光华。然彩凤非无偶，德音始相求。若逐露水之缘，必损九苞之仪。
        解曰：前程似锦，遇合皆良。然姻缘大事，自有天时。此刻宜修己身，待德配之位，慎勿因闲花野草，轻误鸾凤之盟。；
        上上签签文三：
        叁、金鳞遇水
        签文：灵签得「坎」之「比」，卦曰：金鳞耀波，禹门浪翻。乘云气，御飞龙，游乎四海。然渊深鱼乐，网罟斯张。情关如渊，入则难脱。
        解曰：财源广进，机遇良多。然须惕「美人局」，温柔乡是英雄冢。守心如玉，持身如岳，则货利可收而无咎。"
        "中签 · 平和中正
        中签签文一：
        壹、幽兰在谷
        签文：灵签得「艮」之「谦」，卦曰：幽兰生谷，不为莫佩而不芳。舟在江海，不为莫乘而不浮。君子藏器于身，待时而动。然兰生空谷，易招攀折；玉韫荆山，恐有卞和之泣。
        解曰：时机未至，宜守静笃。情缘如风，过耳则逝。若强求缘法，反伤己身。独善其身，可保清宁，静候天时自有春。；
        中签签文二：
        贰、寒潭鹤影
        签文：灵签得「兑」之「困」，卦曰：寒潭渡鹤，形影自怜。清风徐来，乃见本真。言当以诚，行当以慎。然鹤影虽清，双飞则乱；心湖若动，明月难圆。
        解曰：事有阻滞，须防口舌。情之一字，恰如水中观月，雾里看花。执相则迷，不如不触。心若冰清，天塌不惊。；
        中签签文三：
        叁、云鹤巡天
        签文：灵签得「巽」之「小畜」，卦曰：云鹤巡天，孤唳清霄。扶摇九万，志在丹霄。然鸞凤和鸣，非其时也；若恋凡尘莺燕，必损冲霄之翼。
        解曰：志在远方，功业未竟。儿女私情，实为樊笼。此刻当效云鹤，心寄长空，若为情困，则如折翼坠尘，前功尽弃。"
        "下签 · 韬晦待时
        下签签文一：
        壹、潜龙蛰渊
        签文：灵签得「屯」之「益」，卦曰：龙潜于渊，其志难伸。玄黄未判，天地混沌。君子以俭德辟难，不可荣以禄。然深渊之下，暗流汹涌；情丝如网，缚骨缠筋。
        解曰：时运不济，动辄得咎。此刻绝非缔结姻缘之机，否则如龙陷泥沼，愈挣扎愈深。唯有遁世无闷，方能避祸。；
        下签签文二：
        贰、孤舟离岸
        签文：灵签得「未济」之「睽」，卦曰：孤舟离岸，帆楫失利。鬼方未宾，征伐不息。厉终吉，然道孤且艰。况舟小人众，载之则覆；情债累累，偿之则亡。
        解曰：自身难保，何谈风月？此刻如同舟行险滩，任何外缘皆是负累。宜斩断葛藤，轻身向岸，方可于绝处逢生。；
        下签签文三：
        叁、落花辞树
        签文：灵签得「剥」之「坤」，卦曰：繁花辞树，飘零随风。阴霜肃物，芳菲俱尽。小人道长，君子道消。然落红本是无情物，莫向东风怨别离。
        解曰：运势衰颓，恐有分离之象。旧缘如风中残烛，强留无益。新缘更似毒药，饮之伤身。宜彻底放下，清静自守，以待一元复始。"
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
