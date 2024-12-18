from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CodeFile(Base):
    __tablename__ = "code_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    vectors = relationship("VectorEmbedding", back_populates="code_file")

class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    code_file_id = Column(Integer, ForeignKey("code_files.id"), nullable=False)
    dimension_index = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    code_file = relationship("CodeFile", back_populates="vectors")


class Search(Base):
    __tablename__ = "search"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, unique=True, index=True)  # The search query
    result = Column(JSON)  # The raw search results, stored as JSON
    response = Column(String)  # The generated response for the query
    timestamp = Column(Integer)  # Timestamp of when the query was cached

search = Search()
code = CodeFile()
vector = VectorEmbedding()

