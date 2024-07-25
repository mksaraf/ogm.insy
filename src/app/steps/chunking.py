"""

This module defines the chunking of loaded data

"""
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunking(data,chunk_size,chunk_overlap):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks=text_splitter.split_documents(data)
    return chunks   