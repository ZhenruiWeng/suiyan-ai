import os

from sqlalchemy.testing.suite.test_reflection import metadata

import config_data as config
import hashlib
from datetime import datetime

def check_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w').close()   #打开的同时会创建
        return False
    else:
        for line in open(config.md5_path, 'r',encoding='utf-8'):
            line = line.strip()  #处理前后空格与回车
            if line==md5_str:
                return True
        return False

def save_md5(md5_str:str):
    """将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path, 'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')

def get_string_md5(input_str:str,encoding='utf-8'):
    """将传入的字符串转为md5文字"""
    #将字符串转为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    #创建md5对象
    md5 = hashlib.md5()
    md5.update(str_bytes)
    return md5.hexdigest()   #十六进制字符串

#=====================================================================================

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在就创建，存在就跳过
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma=Chroma(
            collection_name=config.collection_name, #数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory #数据库本地存储文件夹
        )  #向量储存实例 Chroma向量库对象

        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       #分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,#连续文本段之间的字符重叠数量
            separators=config.separators,
            length_function=len,
        )


    def upload_by_str(self,data,filename):
        """将传入的字符串向量化，传入向量数据库"""
        md5_hex=get_string_md5(data)

        #查重：
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        #检查是否需要分割：
        if len(data)>config.max_split_char_number:
            knowledge_chunks: list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks:list[str]=[data]

        #元数据定义
        metadata={"source":filename,"create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "operator":"Operator"}

        #将内容加载到向量库
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        #储存md5
        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"

