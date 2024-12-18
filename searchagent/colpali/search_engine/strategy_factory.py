from searchagent.colpali.search_engine.indexing_strategies import (
    HNSWIndexingBinaryQuantizationHammingDistance,
)
from searchagent.colpali.search_engine.search_strategies import SearchStrategy


class SearchStrategyFactory:
    _search_strategies = {}

    @classmethod
    def register_strategy(cls, name: str, search_strategy_cls: type):
        cls._search_strategies[name] = search_strategy_cls

    @classmethod
    def create_strategy(cls, name: str) -> SearchStrategy:
        search_strategy_cls = cls._search_strategies.get(name)
        if not search_strategy_cls:
            raise ValueError(f"Unsupported search strategy: {name}")

        return search_strategy_cls()


class IndexingStrategyFactory:
    _indexing_strategies = {}

    @classmethod
    def register_strategy(cls, name: str, indexing_strategy_cls: type):
        cls._indexing_strategies[name] = indexing_strategy_cls

    @classmethod
    def create_strategy(
        cls, name: str
    ) -> HNSWIndexingBinaryQuantizationHammingDistance:
        indexing_strategy_cls = cls._indexing_strategies.get(name)
        if not indexing_strategy_cls:
            raise ValueError(f"Unsupported indexing strategy: {name}")

        return indexing_strategy_cls()
