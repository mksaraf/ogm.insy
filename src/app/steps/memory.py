"""

This module defines the adding of memory 

"""
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


def addMemory():
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
    )
    return memory

# conversational memory
def conversationalMemory():
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    return conversational_memory