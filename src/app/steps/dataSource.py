"""

This module defines how to connect to data source

"""
import os
from langchain.document_loaders import PyPDFDirectoryLoader

#loader=PyPDFDirectoryLoader("/content/drive/MyDrive/Colab Notebooks/DATA/NorthAmerica1/")


def checkDataSource(data_path):
    # Example of how to check if the file exists
    file_path = data_path
    if os.path.exists(file_path):
        response = {"warning message": f"File Exists"}
        return response
        #print("File exists")
    else:
        response = {"warning message": f"File Does not Exist"}
        return response
    
def dataLoad(data_path):
    loader=PyPDFDirectoryLoader(data_path)
    data=loader.load()
    return data   