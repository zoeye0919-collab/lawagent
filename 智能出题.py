import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. 配置区域 ---
# 请确保这里填写了你的 Key
api_key = "sk-d75143a2504f43089e4c20d2db3a3a52"
os.environ["DASHSCOPE_API_KEY"] = api_key

EXAM_DATA_DIR = "./法考真题"  # 题库源文件目录
EXAM_DB_FILE = "exam_db.json"  # 输出文件

# 初始化模型
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max",
    temperature=0.1,
)


# --- 2. 定义数据结构 ---
class ExamQuestion(BaseModel):
    subject: str = Field(description="题目所属的法律学科，必须从以下列表中选择一个：刑法, 民法, 行政法, 刑事诉讼法, 民事诉讼法, 商经法, 理论法, 三国法")
    question_text: str = Field(description="题目完整的题干内容")
    options: List[str] = Field(description="选项列表，例如 ['A. ...', 'B. ...']")
    correct_answer: Optional[str] = Field(description="正确答案，例如 'A' 或 'ABCD'，如果文中没有则留空")
    analysis: Optional[str] = Field(description="题目解析，如果文中没有则留空")


class QuestionList(BaseModel):
    questions: List[ExamQuestion]


# --- 3. 核心提取逻辑 ---
def extract_questions_from_file(file_path):
    print(f"📄 正在处理文件: {os.path.basename(file_path)} ...")
    ext = os.path.splitext(file_path)[1].lower()

    loader = None
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        print(f"⚠️ 跳过不支持的文件格式: {file_path}")
        return []

    try:
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

        # 注意：此处截取前15000字。如果文档很长，建议在生产环境中做切片循环
        text_chunk = full_text[:15000]

        parser = JsonOutputParser(pydantic_object=QuestionList)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的法考数据处理专家。请从文档中提取真题，并为每一道题进行学科分类。
                    请严格遵守以下规则：
                    1. **自动分类**：根据题目内容，判断其属于以下哪个学科：[刑法, 民法, 行政法, 刑事诉讼法, 民事诉讼法, 商经法, 理论法, 三国法]。
                    2. **格式规范**：输出标准的 JSON 格式。
                    3. **处理缺失**：如果文档中没有答案或解析，对应字段留空字符串。
                    {format_instructions}"""),
            ("human", "【文档内容片段】:\n{text}")
        ])

        chain = prompt | llm | parser
        result = chain.invoke({
            "text": text_chunk,
            "format_instructions": parser.get_format_instructions()
        })
        questions = result.get('questions', [])
        print(f"成功提取 {len(questions)} 道题目")
        return questions
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return []


def main():
    if not os.path.exists(EXAM_DATA_DIR):
        os.makedirs(EXAM_DATA_DIR)
        print(f"📁 已创建文件夹 {EXAM_DATA_DIR}，请将 PDF/Word 放入其中后再次运行。")
        return

    files = [f for f in os.listdir(EXAM_DATA_DIR) if f.endswith(('.pdf', '.docx'))]
    if not files:
        print(f"⚠️ {EXAM_DATA_DIR} 文件夹为空，请放入真题文件。")
        return

    global_db = {}

    print(f"🚀 开始构建智能题库，共 {len(files)} 个文件...")

    for f in files:
        full_path = os.path.join(EXAM_DATA_DIR, f)

        # 提取题目
        extracted_qs = extract_questions_from_file(full_path)

        for q in extracted_qs:
            # 以此处提取的 subject 为准，清洗一下可能的格式杂乱
            sub_name = q.get('subject', '综合知识').strip()

            # 简单的归一化处理
            if "刑法" in sub_name:
                sub_name = "刑法"
            elif "民法" in sub_name:
                sub_name = "民法"
            elif "行政" in sub_name:
                sub_name = "行政法"
            elif "刑诉" in sub_name or "刑事诉讼" in sub_name:
                sub_name = "刑事诉讼法"
            elif "民诉" in sub_name or "民事诉讼" in sub_name:
                sub_name = "民事诉讼法"

            if sub_name not in global_db:
                global_db[sub_name] = []

            # 移除 subject 字段本身再存入（可选，为了 JSON 干净）
            # q.pop('subject', None)
            global_db[sub_name].append(q)

    # 保存
    with open(EXAM_DB_FILE, "w", encoding='utf-8') as f:
        json.dump(global_db, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print("\n🎉 题库构建完成！分类统计如下：")
    for sub, qs in global_db.items():
        print(f"   - {sub}: {len(qs)} 题")
    print(f"文件已保存至: {EXAM_DB_FILE}")


if __name__ == "__main__":
    main()