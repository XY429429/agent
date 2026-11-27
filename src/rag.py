from __future__ import annotations

import sys
import pathlib
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

dim=768  #统一维度为768维

#向量化class
class EmbeddinModel:
    def __init__(self,model_name:str = "BAAI/bge-small-zh-v1.5"):
        self.model_name=model_name
        self.model=None
        self.dim=dim

    def _lazy_load_model(self):
        """懒加载模型"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model=SentenceTransformer(self.model_name)
            except:
                raise ImportError("请先安装sentence_transformers: pip install sentence_transformer")
        return self.model


    def encode(self,texts:List[str])->np.ndarray:
        self._lazy_load_model()

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  #归一化
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings

#向量存储和检索
class VectorStore:
    def __init__(self,use_faiss=True):
        self.dim=dim
        self.use_faiss=use_faiss
        self.index=None
        self.vectors:Optional[np.ndarray]=None

    def build(self,vectors:np.ndarray):
        actual_dim = vectors.shape[1]
        print(f"真实维度：{actual_dim}",file=sys.stderr)
        if self.use_faiss:
            try:
                import faiss
                self.index=faiss.IndexFlatIP(actual_dim)
                self.index.add(vectors)
            except ImportError:
                print("未安装faiss，回退至numpy版本",file=sys.stderr)
                self.use_faiss=False
                self.vectors=vectors
        else:
            self.vectors=vectors

    def search(self,query_vector:np.ndarray,top_k:int):
        if self.use_faiss and self.index is not None:
            score,index=self.index.search(query_vector.reshape(1,-1),top_k)
            return score[0],index[0]
        elif self.vectors is not None:
            similarity=np.dot(self.vectors,query_vector)
            top_index=np.argsort(similarity)[::-1][:top_k]
            return similarity[top_index],top_index
        else:
            return np.ndarray([]),np.ndarray([])


class Rerank:
    """基类"""
    def rerank(self,query:str,documents:List[dict],top_k:int):
        raise NotImplementedError

class CrossEncoderReranker(Rerank):
    def __init__(self,model_name:str="BAAI/bge-reranker-v2-m3"):
        self.model_name=model_name
        self.model=None


    def _lazy_load_model(self):
        """懒加载模型"""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model=CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "请先安装sentence_transformers: pip install sentence_transformers"
                )
        return self.model


    def rerank(self,query:str,documents:List[dict],top_k:int)->List[dict]:
        if not documents:
            return []

        self._lazy_load_model()

        pairs=[[query,document["text"]]for document in documents]
        scores=self.model.predict(pairs)

        for doc,score in zip(documents,scores):
            doc["rerank_score"]=float(score)

        documents.sort(key=lambda x:x["rerank_score"],reverse=True)
        return documents[:top_k]


class VectorRetriever:
    """对外接口"""

    def __init__(self,doc_dir:str|Path,chunk_size:int=370,chunk_overlap:int=37,
                 k:int=20,final_k:int=7,embedding_model:str="BAAI/bge-small-zh-v1.5",
                 use_faiss:bool=True,reranker:Optional[Rerank]=None,
                 headers_to_split_on:Optional[List[tuple[str, str]]] = None)->None:
        self.doc_dir=Path(doc_dir)
        self.k=k
        self.final_k=final_k
        self.embedding_model_name=embedding_model
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.use_faiss=use_faiss
        self.reranker=reranker
        self.docs:List[Document]=[]

        """初始化模型"""
        self.embed_model=EmbeddinModel(self.embedding_model_name)
        self.vectorstore=VectorStore(use_faiss=self.use_faiss)
        self.reranker=reranker or CrossEncoderReranker()

        self.headers_to_split_on=headers_to_split_on or [
            ("#","h1"),
            ("##","h2"),
            ("###","h3")
        ]
        self.load_and_build()

    def load_and_build(self):
        header_split=MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        child_split=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)

        for path in self.doc_dir.glob('*.md'):
            text=path.read_text(encoding='utf-8')
            docs=header_split.split_text(text)
            for doc in docs:
                chunks=child_split.split_documents([doc])
                for chunk in chunks:
                    titles=[]
                    title_name=['h1','h2','h3']
                    for levels in title_name:
                        if levels in chunk.metadata:
                            titles.append(levels)
                    title='<'.join(titles)
                    chunk.metadata["title"]=title
                    chunk.metadata["source"]=path.name
                    self.docs.append(chunk)

        if self.docs:
            content=[doc.page_content for doc in self.docs]
            vectors=self.embed_model.encode(content)
            self.vectorstore.build(vectors)
            print("向量索引已建立",file=sys.stderr)

    def search(self,query:str,top_k:Optional[int]=None):
        if not self.docs or (self.vectorstore.vectors is None and self.vectorstore.index is None):
            return []

        query_vector=self.embed_model.encode([query])

        final_k=top_k or self.final_k
        initial_k=max(self.k,final_k)

        scores,indexes=self.vectorstore.search(query_vector=query_vector[0],top_k=initial_k)
        initial_result=[]
        for score,index in zip(scores,indexes):
            doc=self.docs[index]
            initial_result.append(
                {
                    "doc_id": f"{doc.metadata.get('source', 'unknown')}#{index}",
                    "title": doc.metadata.get("title", ""),
                    "vector_score": float(score),
                    "text": doc.page_content,
                    "metadata": {
                        k: v for k, v in doc.metadata.items()
                        if k not in ("title", "source")
                    },
                }
            )

        if self.reranker and self.docs:
            final_result=self.reranker.rerank(query=query,documents=initial_result,top_k=final_k)

        return final_result or []






if __name__=="__main__":
    paths= pathlib.Path(__file__).absolute().parent.parent / "data" / "knowledge"
    xy=VectorRetriever(doc_dir=paths,chunk_size=370,chunk_overlap=37,k=20,final_k=7)
    query="rag实现要点有哪些"
    ans=xy.search(query=query,top_k=3)
    for tep in ans:
        print(tep)



