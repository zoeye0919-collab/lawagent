import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置 Key
api_key = "sk-d75143a2504f43089e4c20d2db3a3a52"
os.environ["DASHSCOPE_API_KEY"] = api_key


def build_term_vector_db():
    pdf_path = "法律术语参考.pdf"
    if not os.path.exists(pdf_path):
        print("❌ 未找到法律术语参考.pdf")
        return

    print("🚀 正在对术语手册进行向量化...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 法律术语通常较短，切片可以小一点以便精准匹配
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    vector_db = FAISS.from_documents(texts, embeddings)

    # 保存到独立目录
    vector_db.save_local("term_faiss_index")
    print("✅ 术语知识库构建成功，保存在 term_faiss_index 文件夹中。")


if __name__ == "__main__":
    build_term_vector_db()