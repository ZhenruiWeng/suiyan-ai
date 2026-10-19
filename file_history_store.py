from langchain_core.chat_history import BaseChatMessageHistory
import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage,message_to_dict,messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id           #会话id
        self.storage_path = storage_path       #文件夹路径
        #完整的文件路径
        self.file_path=os.path.join(self.storage_path,self.session_id)
        #确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)  # 新的和已有的
        # 二进制类型转为字典（类对象写入文件均为二进制）
        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    def messages(self) ->list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                message_data = json.load(f)
                return(messages_from_dict(message_data))
        except FileNotFoundError:
            return []


    def clear(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)




def print_prompt(full_prompt):
    print("=" * 20, full_prompt.to_string(), "=" * 20)
    return full_prompt

