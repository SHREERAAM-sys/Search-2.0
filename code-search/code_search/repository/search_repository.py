from sqlalchemy.orm import Session
from code_search.database.model import Search


class QueryRepository:
    @staticmethod
    def fetch_search_query(db: Session, query: str) -> Search:
        # Check if the query is cached in the search table
        return db.query(Search).filter(Search.query == query).first()

    @staticmethod
    def save_search_query(db: Session, query: str, search_results: list, response: str):
        # Save the search query, result, and response into the Search table
        search_cache = Search(query=query, result=search_results, response=response)
        db.add(search_cache)
        db.commit()
