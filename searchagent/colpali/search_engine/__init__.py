from searchagent.colpali.search_engine.indexing_strategies import (
    HNSWIndexingBinaryQuantizationHammingDistance,
    HNSWIndexingCosineSimilarity,
)
from searchagent.colpali.search_engine.search_strategies import (
    ANNHNSWCosineSimilaritySearchStrategy,
    ANNHNSWHammingSearchStrategy,
    ExactMaxSimSearchStrategy,
)
from searchagent.colpali.search_engine.strategy_factory import (
    IndexingStrategyFactory,
    SearchStrategyFactory,
)

SearchStrategyFactory.register_strategy(
    "ExactMaxSim",
    ExactMaxSimSearchStrategy,
)

SearchStrategyFactory.register_strategy(
    "ANNHNSWHamming",
    ANNHNSWHammingSearchStrategy,
)

SearchStrategyFactory.register_strategy(
    "ANNHNSWCosineSimilarity",
    ANNHNSWCosineSimilaritySearchStrategy,
)


IndexingStrategyFactory.register_strategy(
    "HNSWBQHamming", HNSWIndexingBinaryQuantizationHammingDistance
)

IndexingStrategyFactory.register_strategy(
    "HNSWCosineSimilarity", HNSWIndexingCosineSimilarity
)
