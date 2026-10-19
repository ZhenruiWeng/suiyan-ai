from PyPDF2 import PdfReader
import streamlit as st
from knowledge_base import KnowledgeBaseService

#添加网页标题
st.title('知识库更新服务')
#file_uploader
uploader_file=st.file_uploader(
    "请上传文件",
    type=["pdf", "png", "jpg", "jpeg", "gif"],
    accept_multiple_files=False,
)
#session_state
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    #提取文件信息
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    #text=uploader_file.getvalue().decode("utf-8")
    #st.write(text)
    # 判断如果是PDF，提取文本
    if file_type == "application/pdf":
        reader = PdfReader(uploader_file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text

        st.success(" PDF文本提取完成")
        with st.expander("预览提取的文本"):
            st.write(full_text)

        result=st.session_state["service"].upload_by_str(full_text, file_name)
        st.write(result)