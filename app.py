import streamlit as st
import os
import tempfile
import time
import json
import random
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_community.tools import DuckDuckGoSearchRun


# 初始化配置

api_key = "sk-d75143a2504f43089e4c20d2db3a3a52"
os.environ["DASHSCOPE_API_KEY"] = api_key

EXAM_DATA_DIR = "./法考真题"
EXAM_DB_FILE = "exam_db.json"

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max",
    temperature=0.1,
)

embeddings = DashScopeEmbeddings(model="text-embedding-v4")

# 初始化搜索工具
search_tool = DuckDuckGoSearchRun()


# --- 2. 加载知识库 ---
# 法律条文知识库
@st.cache_resource
def load_db():
    if os.path.exists("legal_faiss_index"):
        return FAISS.load_local("legal_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None


vectorstore = load_db()

# 法律英文术语知识库
@st.cache_resource
def load_term_db():
    if os.path.exists("term_faiss_index"):
        return FAISS.load_local("term_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

term_vectorstore = load_term_db()

# --- 3. 核心功能函数 ---
# 功能一：法律文本翻译
def legal_translation(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名精通中国法律与普通法系的资深翻译专家。请将用户输入的文本进行专业法律翻译。\n"
                   "- 如果是中文，译为英文。\n"
                   "- 如果是英文，译为中文。\n"
                   "- 重点：确保'Consideration(对价)', 'Performance(履行)', 'Third Party(第三人)'等术语准确。"),
        ("human", "{input}")
    ])
    chain = prompt | llm
    return chain.invoke({"input": text}).content

# 功能二：案例分析
def case_analysis(case_text):
    context_text = ""
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(case_text)
        context_text = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名高级法律顾问。请根据用户提供的【案情事实】，结合【参考资料】（如果有），撰写一份结构化的分析报告。\n"
                   "对案件中的每个争议焦点分别分析，要有对应的法条适用、结论建议。\n"
                   "仅围绕案情中客观呈现的事实进行法律评价，不得擅自补充或假设不存在的事实。\n"
                   "报告格式要求：\n"
                   "1. **案件背景**：对用户上传的案件做简单概括，200-300字左右。\n"
                   "2. **争议焦点**：归纳核心法律问题，要求点明构成什么侵权行为/构成什么罪名。\n"
                   "3. **法条适用**：引用最相关的法律条款，给出法条出处。\n"
                   "4. **结论建议**：给出具体的操作建议。\n "
                   "示例：\n "
                   "1.**案件背景**：某房地产开发项目位于我国某一线城市，开发商为某房地产开发有限公司(以下简称“开发商”)，项目名为“某湾花园”该项目占地约1000亩，总建筑面积约200万平方米，\n "
                   "包括住宅、商业、办公等多种业态，项目自2008年开始建设，预计嵯部面等13年竣工。然而，在项目施工过程中，开发商与部分业主就房屋质量问题产生了纠纷，进面引发了诉讼。\n"
                   "2.**争议焦点**：1.房屋质量问题 \n业主认为，房屋存在以下质量问题:\n(1)墙体裂缝:业主反映，部分墙体出现裂缝，裂缝长度不一，宽度从几毫米到几厘米不等(2)渗水问题:业主反映，部分房屋存在渗水现象，尤其在雨天更为严重。\n"
                   "3.**法条适用**：1.房屋质量问题:关于房屋质量问题，根据《中华人民共和国建筑法》和《建设工程质量管理条例》的相关规定，开发商应保证房屋质量符唖合国家标准，本案中，业主反映的墙体裂缝、渗水等问题、经鉴定，确属房屋质量问题。对此，开发商应承担相应的法律责任。\n"
                   "4.**结论建议**：1.对业主反映的房原质量问题，开发商应负责修复，修复费用由开发商承担。"),
        ("human", "【案情事实】：{input}\n\n【参考资料】：{context}")
    ])
    chain = prompt | llm
    return chain.invoke({"input": case_text, "context": context_text}).content

# 功能三：法律问答
def smart_qa_search(query, vectorstore):
    if not vectorstore:
        return run_web_search(query)

    # 1. 检索本地知识
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    # 2. 构造带来源的上下文
    context_items = []
    for i, d in enumerate(docs):
        source = os.path.basename(d.metadata.get("source", "未知法条"))
        context_items.append(f"【条文依据 {i+1}】(出处: {source}):\n{d.page_content}")

    context_text = "\n\n".join(context_items)

    # 3. 强化法律学习与推理的提示词
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名极其严谨的中国法律专家。请基于提供的【法律条文依据】回答用户问题，并加入自己的推理思考与理解。
        
请遵循以下回答准则：
1. **法条优先**：必须优先使用提供的条文内容。回答时请明确指出引用了哪一条依据。
2. **逻辑推导**：不要简单复制法条，要解释该法条如何适用于用户的具体问题。
3. **严谨性**：如果提供的【法律条文依据】中完全没有相关信息，请直接回答 'SEARCH_NEEDED'。
4. **法律用语**：使用专业、客观的法律术语，避免口语化。
5. **法律伦理**：触犯伦理道德、公民隐私、违反宪法的问题禁止回答。

【法律条文依据】：
{context}"""),
        ("human", "{input}")
    ])

    chain = qa_prompt | llm
    response = chain.invoke({"context": context_text, "input": query}).content

    # 4. 判断是否需要联网补充
    if "SEARCH_NEEDED" in response:
        return run_web_search(query)
    else:
        # 格式化输出，增强“已学习本地知识”的感知
        final_answer = f"{response}\n\n---\n** 本次回答基于以下法律条文：**\n"
        sources = list(set([os.path.basename(d.metadata.get("source", "法律文档")) for d in docs]))
        for s in sources:
            final_answer += f"- {s}\n"
        return final_answer


def run_web_search(query):
    """执行联网搜索并总结"""
    try:
        # 1. 执行搜索
        search_results = search_tool.invoke(query)

        # 2. 让 AI 总结搜索结果
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一名助手。用户的问题在本地文档中未找到答案，系统已自动联网搜索。请根据以下搜索结果回答用户问题。"),
            ("human", "【搜索结果】：\n{results}\n\n【用户问题】：{query}")
        ])
        chain = summary_prompt | llm
        answer = chain.invoke({"results": search_results, "query": query}).content

        return f"{answer}\n\n---\n**🌐 答案来源：联网检索**"
    except Exception as e:
        return f"⚠️ 本地文档未找到答案，且联网搜索失败：{e}"


# 功能五：文献阅读
# 1. 文档加载器
def load_document(file_path):
    """根据文件后缀自动选择加载器 (支持 PDF, DOCX, TXT)"""
    if file_path.endswith(".txt"):
        # encoding="utf-8" 防止中文乱码
        return TextLoader(file_path, encoding="utf-8").load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

# 2. 文档处理器
def process_document(uploaded_file):
    """
    处理上传文件：保存临时文件 -> 加载 -> 摘要 -> 向量化(DashScope)
    """
    # 1. 获取文件后缀并保存
    # uploaded_file.name 获取文件名，splitext 分离后缀
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext not in [".pdf", ".docx", ".txt"]:
        raise ValueError("仅支持 .pdf, .docx, .txt 格式")

    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. 调用统一加载逻辑
        docs = load_document(tmp_path)
        # 3. 生成摘要 (截取前15k字符以防超长)
        full_text = "\n\n".join([d.page_content for d in docs])
        if not full_text.strip():
            raise ValueError("文档内容为空或无法识别")
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一名学术助手。请阅读以下文献内容，生成一份结构化摘要。\n"
                       "要求包含：\n"
                       "1. 核心观点 (Core Argument)\n"
                       "2. 主要论据/方法 (Methodology)\n"
                       "3. 研究结论 (Conclusion)\n"
                       "4. 创新点或局限性 (若有)\n"
                       "字数控制在 600 字以内。"),
            ("human", "【文献内容片段】\n{text}")
        ])
        summary = (summary_prompt | llm).invoke({"text": full_text[:15000]}).content
        # 4. 构建专属向量库 (使用全局定义的 embeddings)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        # 直接使用 DashScope 的 embeddings 对象
        vectorstore = FAISS.from_documents(splits, embeddings)
        return summary, vectorstore

    except Exception as e:
        raise e
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



# 3：文献问答
def ask_document_with_reasoning(query, vectorstore):
    """
    文献阅读专用问答函数：
    1. 检索更多上下文 (k=5)
    2. 使用思维链 Prompt 让模型深度思考
    3. 返回答案并附带原文引用
    """
    if not vectorstore:
        return "请先上传文献。"

    # 1. 检索：稍微增加 k 值以获取更多上下文，便于综合分析
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    # 整理上下文，保留页码/来源信息以便引用
    context_parts = []
    for i, d in enumerate(docs):
        # 截取每个片段的前200字展示在来源中，防止过长
        content_preview = d.page_content.replace('\n', ' ')
        source_info = f"[片段{i + 1}] {content_preview}..."
        context_parts.append(d.page_content)

    context_text = "\n\n".join(context_parts)

    # 2. 构建思维链 Prompt (Chain of Thought)
    # 这里的 System Prompt 是让模型"思考"的关键
    system_prompt = """你是一名严谨的学术研究助手。请阅读下方的【参考文献片段】，并回答用户的问题。

请按照以下步骤思考（Chain of Thought）：
1. **信息定位**：在参考文献中找到与问题相关的具体句子或段落。
2. **逻辑分析**：结合上下文理解这些信息的含义，排除无关干扰。
3. **答案生成**：基于原文事实生成答案，不要编造。如果原文中没有提及，请明确说明“文中未提及”。
4. **引用标注**：在回答中适当引用原文的关键表述。

【参考文献片段】：
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 3. 生成回答
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "input": query}).content

    # 4. 格式化输出：答案 + 参考来源
    # 提取来源文件名
    sources = list(set([os.path.basename(d.metadata.get("source", "当前文档")) for d in docs]))
    source_str = "\n".join([f"- {s}" for s in sources])

    # 可以在这里把检索到的具体片段也折叠显示出来，增强“有据可查”的感觉
    detailed_sources = "\n".join([f"> **片段 {i + 1}**: {d.page_content[:100]}..." for i, d in enumerate(docs)])

    final_output = f"{response}\n\n---\n**📚 思考依据**：\n{detailed_sources}"

    return final_output


# 导出历史记录辅助函数
def convert_history_to_md(chat_history, summary=""):
    """对话类历史转Markdown (问答、文献阅读)"""
    md_text = f"# ⚖️ 对话记录存档 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    if summary:
        md_text += f"## 📄 摘要\n{summary}\n\n---\n\n"
    md_text += "## 💬 对话详情\n"
    for msg in chat_history:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        md_text += f"### {role_icon} {msg['role']}:\n{msg['content']}\n\n"
    return md_text


def generate_universal_md(data_list, mode):
    """非对话类历史转Markdown (翻译、案例、法考)"""
    md = f"# ⚖️ {mode}记录存档 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for idx, item in enumerate(data_list, 1):
        md += f"## 记录 {idx}\n"
        if mode == "Translation":
            md += f"**原文**:\n{item['input']}\n\n**译文**:\n{item['output']}\n"
        elif mode == "CaseAnalysis":
            md += f"**案情**:\n{item['input']}\n\n**分析报告**:\n{item['output']}\n"
        elif mode == "Exam":
            md += f"**科目**: {item['subject']}\n**题目**:\n{item['q']}\n\n**解析**:\n{item['a']}\n"
        md += "\n---\n"
    return md


def render_history_ui(session_key, mode_name, md_type="universal"):
    """统一渲染历史记录的下载与清空按钮"""
    if session_key in st.session_state and st.session_state[session_key]:
        st.divider()
        st.caption(f"📊 {mode_name} - 历史记录管理")
        c1, c2 = st.columns([1, 1])
        with c1:
            data = st.session_state[session_key]
            if md_type == "universal":
                # 映射 mode_name 到内部 mode 字符串
                internal_mode = "Translation" if "翻译" in mode_name else "CaseAnalysis" if "案例" in mode_name else "Exam"
                md_text = generate_universal_md(data, internal_mode)
            else:
                # 对话类直接传列表
                md_text = convert_history_to_md(data)

            st.download_button(
                label=f"📥 导出{mode_name}记录",
                data=md_text,
                file_name=f"{mode_name}_History.md",
                mime="text/markdown"
            )
        with c2:
            if st.button(f"🗑️ 清空{mode_name}记录", key=f"clear_{session_key}"):
                st.session_state[session_key] = []
                st.rerun()

        # 简单展示历史条目数
        if md_type == "universal":
            with st.expander(f"查看历史列表 ({len(st.session_state[session_key])}条)"):
                for i, item in enumerate(reversed(st.session_state[session_key])):
                    st.text(f"记录 {len(st.session_state[session_key]) - i}")
                    # 简略显示内容
                    if 'input' in item:
                        st.caption(item['input'][:50] + "...")
                    elif 'q' in item:
                        st.caption(item['q'][:50] + "...")
                    st.divider()

# --- 4. Streamlit 前端界面 ---

st.title("⚖️法助手")

# 侧边栏
MENU_OPTIONS = [
    "法律文本翻译",
    "案例智能分析",
    "法律知识问答",
    "文献阅读",
    "法考备考"
]
option = st.sidebar.radio("功能导航", MENU_OPTIONS)
st.sidebar.markdown("---")

if st.sidebar.button("🗑️ 清空当前会话", type="primary"):
    # 清除所有 Session State
    st.session_state.clear()
    # 强制刷新页面
    st.rerun()

# --- 功能逻辑 ---
# 1. 法律文本翻译
if option == "法律文本翻译":
    st.header("专业法律文本翻译")

    # --- 1. 输入区域选择 ---
    input_method = st.radio("选择输入来源", ["✍️ 手动输入文本", "📄 上传文档 (PDF/Docx)"], horizontal=True)

    final_text = ""  # 存储最终待翻译的内容

    if input_method == "📄 上传文档 (PDF/Docx)":
        uploaded_file = st.file_uploader("请上传待翻译文件", type=["pdf", "docx"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
                final_text = "\n".join([d.page_content for d in docs])
                st.success(f"✅ 文件解析成功，共提取 {len(final_text)} 字")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
    else:
        final_text = st.text_area("请在此粘贴待翻译文本", height=200, placeholder="在此输入...")

    # --- 2. 翻译设置 ---
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        target_lang = st.selectbox("目标语言", ["英文", "中文"])
    with c2:
        use_rag = st.checkbox("启用术语库增强", value=True)

    # --- 3. 执行翻译 ---
    if st.button("开始翻译", type="primary", use_container_width=True):
        if not final_text.strip():
            st.warning("内容为空，请输入文本或上传文件。")
        else:
            with st.spinner("法助手正在检索术语库并生成译文..."):
                # 检索相关术语 (RAG)
                context_terms = ""
                if use_rag and term_vectorstore:
                    # 检索最相关的20条术语对
                    search_docs = term_vectorstore.similarity_search(final_text[:500], k=20)
                    context_terms = "\n".join([d.page_content for d in search_docs])

                # 构建翻译链
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一位资深的法律翻译专家。请将以下文本翻译为{lang}。

                    【翻译准则】：
                    1. 参考提供的【术语对照表】，确保核心词汇专业且统一。
                    2. 使用正式法律文体，保持条款编号和格式。

                    【术语对照表】：
                    {context}"""),
                    ("human", "待翻译文本：\n{text}")
                ])

                chain = prompt | llm
                res = chain.invoke({
                    "lang": target_lang,
                    "context": context_terms,
                    "text": final_text
                })

                st.subheader("📑 翻译结果")
                st.success(res.content)

                st.download_button("下载译文", res.content, file_name="translated_legal.txt")


# 2. 案例智能分析
elif option == "案例智能分析":
    st.header("案例案情分析")

    if "case_history" not in st.session_state:
        st.session_state.case_history = []

    case_input = st.text_area("请输入案情事实", height=200)

    if st.button("生成分析报告"):
        if not case_input.strip():
            st.warning("请输入案情")
        else:
            with st.spinner("撰写报告..."):
                # 优化：只调用一次 AI
                res = case_analysis(case_input)
                st.markdown(res)
                st.session_state.case_history.append({"input": case_input, "output": res})

    # 历史记录
    render_history_ui("case_history", "案例分析")


# 3. 法律知识问答
elif option == "法律知识问答":
    st.header("法律知识问答")
    st.markdown("法助手会结合本地知识库回答你的提问。")

    with st.form(key="qa_form", clear_on_submit=True):  # clear_on_submit=True 发送后清空
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("问题", placeholder="请输入你的问题...", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("发送")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史记录
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 逻辑判断：如果点击了提交按钮 且 输入框不为空
    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                full_res = smart_qa_search(user_input, vectorstore)
                st.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})

    # 历史管理
    if st.session_state.messages:
        render_history_ui("messages", "问答", md_type="chat")


# 4. 文献阅读
elif option == "文献阅读":
    st.header("文献智能阅读 & 深度思考")

    if "doc_state" not in st.session_state:
        st.session_state.doc_state = {
            "current_file_name": None, "summary": "", "vectorstore": None, "chat_history": []
        }

    # 1. 文件上传
    uploaded_file = st.file_uploader("📂 上传文献 (PDF/Docx/Txt)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        current_name = uploaded_file.name
        # 只有文件名变化时才重新解析
        if current_name != st.session_state.doc_state["current_file_name"]:
            with st.status("🔍 正在深度阅读文献...", expanded=True) as status:
                try:
                    st.write("1. 正在提取文本内容...")
                    # 调用之前的 process_document 函数
                    summary, vs = process_document(uploaded_file)

                    st.write("2. 正在构建知识索引...")
                    st.write("3. 正在生成全文摘要...")

                    # 更新状态
                    st.session_state.doc_state.update({
                        "current_file_name": current_name,
                        "summary": summary,
                        "vectorstore": vs,
                        "chat_history": []
                    })
                    status.update(label="✅ 文献阅读完成！", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"解析失败: {e}")
                    status.update(label="❌ 失败", state="error")

    # 2. 显示摘要
    if st.session_state.doc_state["summary"]:
        with st.expander("📄 点击查看【全文智能摘要】", expanded=True):
            st.markdown(st.session_state.doc_state["summary"])

    # 3. 对话区域
    if st.session_state.doc_state["vectorstore"]:
        st.divider()
        st.subheader("💬 基于文献的问答助手")

        # 渲染历史记录
        for msg in st.session_state.doc_state["chat_history"]:
            st.chat_message(msg["role"]).write(msg["content"])

        # 输入框
        if query := st.chat_input("向 AI 提问关于这篇文献的内容..."):
            # 用户消息
            st.session_state.doc_state["chat_history"].append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            # AI 回答
            with st.chat_message("assistant"):
                with st.spinner("🤔 正在检索原文并思考..."):
                    # === 关键修改：调用带思考逻辑的专用函数 ===
                    answer = ask_document_with_reasoning(query, st.session_state.doc_state["vectorstore"])

                    st.markdown(answer)
                    st.session_state.doc_state["chat_history"].append({"role": "assistant", "content": answer})

        # 底部功能栏
        if st.session_state.doc_state["chat_history"]:
            st.divider()
            c1, c2 = st.columns([1, 1])
            with c1:
                # 导出功能
                md = convert_history_to_md(
                    st.session_state.doc_state["chat_history"],
                    st.session_state.doc_state["summary"]
                )
                st.download_button("📥 导出本篇对话记录", md, "doc_reading.md")
            with c2:
                # 清空功能
                if st.button("🗑️ 清空本篇对话"):
                    st.session_state.doc_state["chat_history"] = []
                    st.rerun()

# 5. 法考备考
elif option == "法考备考":
    st.header("📚 法考智能刷题系统")

    # 1. 加载题库
    if "exam_db" not in st.session_state:
        if os.path.exists(EXAM_DB_FILE):
            with open(EXAM_DB_FILE, "r", encoding="utf-8") as f:
                st.session_state.exam_db = json.load(f)
        else:
            st.session_state.exam_db = {}

    if not st.session_state.exam_db:
        st.error("未检测到本地题库文件 (exam_db.json)。")
        st.info("请确保已运行 `智能出题.py` 生成题库。")
    else:
        # --- 初始化全局 Session 变量 ---
        if "current_q_index" not in st.session_state: st.session_state.current_q_index = 0
        if "show_exam_answer" not in st.session_state: st.session_state.show_exam_answer = False
        if "ai_exam_analysis" not in st.session_state: st.session_state.ai_exam_analysis = None
        if "exam_history" not in st.session_state: st.session_state.exam_history = []
        if "last_subject" not in st.session_state: st.session_state.last_subject = None

        # 2. 科目选择
        subjects = list(st.session_state.exam_db.keys())
        selected_sub = st.selectbox("选择练习科目", subjects)

        # 如果切换了科目，自动重置状态
        if st.session_state.last_subject != selected_sub:
            st.session_state.current_q_index = 0
            st.session_state.show_exam_answer = False
            st.session_state.ai_exam_analysis = None
            st.session_state.last_subject = selected_sub

        question_pool = st.session_state.exam_db[selected_sub]

        # 3. 题目控制工具栏
        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.caption(f"当前科目：**{selected_sub}** | 题库量：{len(question_pool)}")
        with t_col2:
            if st.button("🎲 随机抽题", use_container_width=True):
                # 确保随机抽到的不是当前这一题（如果题库大于1道的话）
                new_idx = st.session_state.current_q_index
                if len(question_pool) > 1:
                    while new_idx == st.session_state.current_q_index:
                        new_idx = random.randint(0, len(question_pool) - 1)
                st.session_state.current_q_index = new_idx
                st.session_state.show_exam_answer = False
                st.session_state.ai_exam_analysis = None
                st.rerun()

        # 4. 题目渲染区域
        if question_pool:
            q_data = question_pool[st.session_state.current_q_index]

            with st.container(border=True):
                st.subheader(f"题目 {st.session_state.current_q_index + 1}")
                st.markdown(f"**{q_data['question_text']}**")

                options = q_data.get('options', [])
                user_choice = None

                if options:
                    # 使用特定的 Key 确保单选框随题目索引刷新
                    user_choice = st.radio(
                        "请选择你的答案：",
                        options,
                        index=None,
                        key=f"radio_{selected_sub}_{st.session_state.current_q_index}"
                    )
                else:
                    st.info("此题为非选择题（主观题/判断题）。")
                    user_choice = st.text_area("答题思路记录：",
                                               key=f"text_{selected_sub}_{st.session_state.current_q_index}")

                st.write("")
                if st.button("提交答案", type="primary"):
                    if (options and len(options) > 0) and not user_choice:
                        st.warning("请先选择一个选项！")
                    else:
                        st.session_state.show_exam_answer = True

                        std_answer = q_data.get('correct_answer')
                        raw_analysis = q_data.get('analysis', "").strip()

                        # 情况 1：数据库中完全没有答案
                        if not std_answer or not str(std_answer).strip():
                            with st.spinner("原真题未包含答案，法助手正在检索本地法条库进行深度解析..."):
                                query = f"题目：{q_data['question_text']}\n选项：{options}\n请给出正确答案和详细法律解析。"
                                ai_res = smart_qa_search(query, vectorstore)
                                st.session_state.ai_exam_analysis = ai_res

                        # 情况 2：有答案但【解析为空】（核心改进点）
                        elif not raw_analysis:
                            with st.spinner("检测到解析缺失，正在根据答案生成专业解析..."):
                                complement_prompt = ChatPromptTemplate.from_messages([
                                    ("system",
                                     "你是一名资深法考讲师。用户会给你一道题目、选项以及标准答案，请你结合中国现行法律条文，给出准确、详尽的法理分析。"),
                                    ("human", "【题目】：{question}\n【选项】：{options}\n【标准答案】：{answer}")
                                ])
                                chain = complement_prompt | llm
                                ai_res = chain.invoke({
                                    "question": q_data['question_text'],
                                    "options": options,
                                    "answer": std_answer
                                }).content
                                st.session_state.ai_exam_analysis = ai_res

                        # 情况 3：已有完整解析
                        else:
                            st.session_state.ai_exam_analysis = None

                        # 统一记录到历史记录中
                        final_a = st.session_state.ai_exam_analysis if st.session_state.ai_exam_analysis else raw_analysis
                        st.session_state.exam_history.append({
                            "subject": selected_sub,
                            "q": q_data['question_text'],
                            "a": f"标准答案：{std_answer}\n解析：{final_a}"
                        })

                    # --- 修改后的显示区域 ---
                if st.session_state.show_exam_answer:
                    st.divider()
                    st.markdown("### 💡 答案解析")

                    std_answer = q_data.get('correct_answer')

                    # 判定用户选择对错
                    if std_answer and user_choice:
                        if str(std_answer) in str(user_choice):
                            st.success("🎉 回答正确！")
                        else:
                            st.error(f"❌ 回答错误。标准答案是：{std_answer}")

                    # 优先显示 AI 补全的解析，如果没有则显示原始解析
                    if st.session_state.ai_exam_analysis:
                        st.info("**【法助手深度解析】**")
                        st.markdown(st.session_state.ai_exam_analysis)
                    else:
                        st.info("**【真题解析】**")
                        st.markdown(q_data.get('analysis') or "暂无解析")

    render_history_ui("exam_history", "法考练习记录")
