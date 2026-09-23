from config import config
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        # We now use the Enterprise-grade Langchain Splitter (DL-05 standard)
        # It recursively tries to split text logically (by double newlines, then newlines, then periods) 
        # before awkwardly cutting a word in half. Perfect for unstructured internet data!
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split_text(self, text):
        """
        Splits a large string into smaller, semantically intact chunks.
        """
        return self.splitter.split_text(text)
