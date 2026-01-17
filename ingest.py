import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 配置 API KEY (建议保持最新)
api_key = "sk-d75143a2504f43089e4c20d2db3a3a52"
os.environ["DASHSCOPE_API_KEY"] = api_key


def create_vector_db():
    # 1. 设置本地数据目录
    data_path = './data'

    print(f"正在扫描 {data_path} 目录下的法律文档...")

    # 2. 分别加载不同格式的文档
    # 加载 Word 文档
    docx_loader = DirectoryLoader(data_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    # 加载 PDF 文档 (使用 PyPDFLoader)
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)

    try:
        docx_documents = docx_loader.load()
        pdf_documents = pdf_loader.load()
        documents = docx_documents + pdf_documents
    except Exception as e:
        print(f"加载过程中出错: {e}")
        return

    if not documents:
        print("Error: 未能在目录中找到任何 .docx 或 .pdf 文件！")
        return

    print(f"成功提取: {len(docx_documents)} 份 Word, {len(pdf_documents)} 份 PDF，共计 {len(documents)} 份文档。")

    # 3. 针对法律条文的切分策略
    # 法律条文逻辑紧密，chunk_size不宜过小，确保法条完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", "！", "？"]
    )
    texts = text_splitter.split_documents(documents)

    print(f"切分完成：生成 {len(texts)} 个知识片段。")
    print("正在连接DashScope进行向量化 (模型：text-embedding-v4)...")

    # 4. 初始化升级后的 text-embedding-v4 模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")

    # 5. 构建并保存本地索引
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local("legal_faiss_index")
    print("向量数据库构建成功！已保存至：legal_faiss_index")


if __name__ == "__main__":
    create_vector_db()