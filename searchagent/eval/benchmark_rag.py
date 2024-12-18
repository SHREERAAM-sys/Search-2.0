"""
Adapted from https://huggingface.co/learn/cookbook/rag_evaluation.

This code demonstrates common practices today for prototyping a RAG system,
which will serve as a baseline for developing a production-ready version.
We will apply the same evaluation metrics from this prototype to the production system.

Synthetic dataset generation process:
1. An LLM generates the evaluation dataset, with questions filtered by other LLMs.
2. An LLM-as-a-judge agent evaluates the dataset, keeping only those that score above 4.

Benchmark Dataset:
- n_generation: Defines the number of QA pairs to generate (final dataset will be smaller).
- Source: https://huggingface.co/datasets/m-ric/huggingface_doc.
- Model: llama3.1:8b-instruct-fp16.
- Text chunking:
    - RecursiveCharacterTextSplitter
    - chunk_size: 2000, chunk_overlap: 200
    - separators: ["\n\n", "\n", ".", " ", ""]
    - Answers < 300 characters.

Evaluation metrics:
- Groundness
- Relevance
- Standalone


The baseline RAG system is configured with:
- FAISS for efficient similarity search and clustering of dense vectors
- Embedding Model: "thenlper/gte-small"
- Reranker Model: "colbert-ir/colbertv2.0" via RAGatouille
- Answer Evaluation: Uses llama3.1:8b-instruct-fp16 to assess RAG-generated answers
- Chunking: Chunk size of 200 with an overlap of 20
- Evaluation Dataset: https://huggingface.co/datasets/friedahuang/m-ric_huggingface_doc_347
"""

import json
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import ollama
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragatouille import RAGPretrainedModel
from searchagent.eval.prompts.qa_generation_prompt import QA_generation_prompt
from searchagent.eval.prompts.question_critique_prompt import (
    question_groundedness_critique_prompt,
    question_relevance_critique_prompt,
    question_standalone_critique_prompt,
)
from searchagent.utils import project_paths
from tqdm.auto import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*"
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*GradScaler.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*CUDA is not available.*"
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")


load_dotenv()


class QAGenerationEvaluator:

    def __init__(
        self,
        llm_model_name="llama3.1:8b-instruct-fp16",
        n_generations=2000,
        dataset_name="m-ric/huggingface_doc",
        embedding_model_name="thenlper/gte-small",
        reranker_model_name="colbert-ir/colbertv2.0",
    ):

        self.llm_model_name = llm_model_name
        self.n_generations = n_generations
        self.dataset_name = dataset_name
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.ds = datasets.load_dataset(dataset_name, split="train")
        self.langchain_docs = [
            LangchainDocument(
                page_content=doc["text"], metadata={"source": doc["source"]}
            )
            for doc in tqdm(self.ds)
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.docs_processed = self._process_documents()
        self.outputs = None
        self.generated_questions = None

    # GENERATE SYNTHETIC DATASET
    def _process_documents(self) -> List:
        docs_processed = []
        for doc in self.langchain_docs:
            docs_processed += self.text_splitter.split_documents([doc])
        return docs_processed

    def call_llm(self, prompt: str) -> str:
        res = ollama.generate(
            model=self.llm_model_name,
            prompt=prompt,
            options={"num_predict": 1000},
        )
        return res["response"]

    def generate_qa_pairs(self) -> List[Dict]:
        outputs = []
        for sampled_context in tqdm(
            random.sample(self.docs_processed, self.n_generations)
        ):
            # Generate QA couple
            output_QA_couple = self.call_llm(
                QA_generation_prompt.format(context=sampled_context.page_content)
            )
            try:
                question = output_QA_couple.split("Factoid question: ")[-1].split(
                    "Answer: "
                )[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except Exception:
                continue

        return outputs

    def critique_qa_pairs(self):
        self.outputs = self.generate_qa_pairs()
        for output in tqdm(self.outputs):
            evaluations = {
                "groundedness": self.call_llm(
                    question_groundedness_critique_prompt.format(
                        context=output["context"], question=output["question"]
                    ),
                ),
                "relevance": self.call_llm(
                    question_relevance_critique_prompt.format(
                        question=output["question"]
                    ),
                ),
                "standalone": self.call_llm(
                    question_standalone_critique_prompt.format(
                        question=output["question"]
                    ),
                ),
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    output.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception:
                continue
        return self

    def filter_qa_pairs(self):
        generated_questions = pd.DataFrame.from_dict(self.outputs)
        self.generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= 4)
            & (generated_questions["relevance_score"] >= 4)
            & (generated_questions["standalone_score"] >= 4)
        ]
        return self

    def create_eval_dataset(self) -> Dataset:
        return datasets.Dataset.from_pandas(
            self.generated_questions, split="train", preserve_index=False
        )

    def evaluate_and_filter_qa_pairs(self):
        return self.critique_qa_pairs().filter_qa_pairs().create_eval_dataset()

    def save_synthetic_test_data(self):
        output_filename = (
            f"{self.dataset_name.replace('/', '~')}_{len(self.generated_questions)}.csv"
        )
        df = self.evaluate_and_filter_qa_pairs()
        OUTPUT_TESTDATA_DIR = project_paths.DATA / "synthetic_testdata"

        if not os.path.exists(OUTPUT_TESTDATA_DIR):
            os.mkdir(OUTPUT_TESTDATA_DIR)

        df.to_csv(OUTPUT_TESTDATA_DIR / output_filename)

    @staticmethod
    def run_rag_tests(
        answer_with_rag: Callable[
            [str, VectorStore, Optional[Any]],
            Tuple[str, List[LangchainDocument]],
        ],
        eval_dataset: datasets.Dataset,
        knowledge_index: VectorStore,
        output_file: str,
        reranker: Optional[Any] = None,
        verbose: Optional[bool] = True,
        test_settings: Optional[str] = None,  # To document the test settings used
    ):
        """Runs RAG tests on the given dataset and saves the results to the given output file."""
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        outputs = []

        for example in tqdm(eval_dataset):
            question = example["question"]
            if question in [output["question"] for output in outputs]:
                continue

            answer, relevant_docs = answer_with_rag(
                question, knowledge_index, reranker=reranker
            )
            if verbose:
                print("=======================================================")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')
            result = {
                "question": question,
                "true_answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_answer": answer,
                "retrieved_docs": [doc for doc in relevant_docs],
            }
            if test_settings:
                result["test_settings"] = test_settings
            outputs.append(result)

            with open(output_file, "w") as f:
                json.dump(outputs, f)

    # BENCHMARK RAG SYSTEM
    @property
    def rag_prompt_template(self):
        return """
        <|system|>
        Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.</s>
        <|user|>
        Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}
        </s>
        <|assistant|>
        """

    @property
    def evaluation_prompt_template(self):
        return """###Task Description:
        An instruction (might include an Input inside it), a response to evaluate,
        a reference answer that gets a score of 5,
        and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly
        based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5.
        You should refer to the score rubric.
        3. The output format should look as follows:
        \"Feedback: {{write a feedback for criteria}}
        [RESULT] {{an integer number between 1 and 5}}\"
        4. Please do not generate any other opening, closing, and explanations.
        Be sure to include [RESULT] in your output.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        {reference_answer}

        ###Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 1: The response is completely incorrect, inaccurate, and/or not factual.
        Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
        Score 3: The response is somewhat correct, accurate, and/or factual.
        Score 4: The response is mostly correct, accurate, and factual.
        Score 5: The response is completely correct, accurate, and factual.

        ###Feedback:"""

    def load_fake_knowledge_base(self):
        return [
            LangchainDocument(
                page_content=doc["text"], metadata={"source": doc["source"]}
            )
            for doc in tqdm(self.ds)
        ]

    @staticmethod
    def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: str,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of size `chunk_size` characters and return a list of documents.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name).to(device),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    def load_embeddings(
        self,
        langchain_docs: List[LangchainDocument],
        chunk_size: int,
    ) -> FAISS:
        """
        Creates a FAISS index from the given embedding model and documents.
        Loads the index directly if it already exists.

        Args:
            langchain_docs: list of documents
            chunk_size: size of the chunks to split the documents into
            embedding_model_name: name of the embedding model to use

        Returns:
            FAISS index
        """
        # load embedding_model
        embedding_model_name = self.embedding_model_name
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=True,
            encode_kwargs={
                "normalize_embeddings": True
            },  # set True to compute cosine similarity
        )

        # Check if embeddings already exist on disk
        index_name = f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
        index_folder_path = f"./data/indexes/{index_name}/"
        if os.path.isdir(index_folder_path):
            return FAISS.load_local(
                index_folder_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )

        else:
            print("Index not found, generating it...")
            docs_processed = __class__.split_documents(
                chunk_size,
                langchain_docs,
                embedding_model_name,
            )
            knowledge_index = FAISS.from_documents(
                docs_processed,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            knowledge_index.save_local(index_folder_path)
            return knowledge_index

    def answer_with_rag(
        self,
        question: str,
        knowledge_index: VectorStore,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 7,
    ) -> Tuple[str, List[LangchainDocument]]:
        """Answer a question using RAG with the given knowledge index."""
        # Gather documents with retriever
        relevant_docs = knowledge_index.similarity_search(
            query=question, k=num_retrieved_docs
        )
        relevant_docs = [
            doc.page_content for doc in relevant_docs
        ]  # keep only the text

        # Optionally rerank results
        if reranker:
            relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
        )

        final_prompt = self.rag_prompt_template.format(
            question=question, context=context
        )

        # Redact an answer
        answer = self.call_llm(final_prompt)

        return answer, relevant_docs

    @staticmethod
    def evaluate_answers(
        answer_path: str,
        eval_chat_model,
        evaluator_name: str,
        evaluation_prompt_template: ChatPromptTemplate,
    ) -> None:
        """Evaluates generated answers. Modifies the given answer file
        in place for better checkpointing."""
        answers = []
        if os.path.isfile(answer_path):  # load previous generations if they exist
            answers = json.load(open(answer_path, "r"))

        for experiment in tqdm(answers):
            eval_prompt = evaluation_prompt_template.format_messages(
                instruction=experiment["question"],
                response=experiment["generated_answer"],
                reference_answer=experiment["true_answer"],
            )
            eval_result = eval_chat_model.invoke(eval_prompt)
            feedback, score = [
                item.strip() for item in eval_result.content.split("[RESULT]")
            ]
            experiment[f"eval_score_{evaluator_name}"] = score
            experiment[f"eval_feedback_{evaluator_name}"] = feedback

            with open(answer_path, "w") as f:
                json.dump(answers, f)

    def run_evals(self):
        chunk_size = 200
        embeddings = self.embedding_model_name
        llm_model = self.llm_model_name
        reranker_model = self.reranker_model_name
        eval_dataset = "friedahuang/m-ric_huggingface_doc_347"

        settings_name = (
            f"chunk{chunk_size}_"
            f"embeddings_{embeddings.replace('/', '~')}_"
            f"rerank_{reranker_model.replace('/', '~')}_"
            f"reader-model_{llm_model.replace(':', '-')}"
        )
        output_file_name = f"./eval_output/rag_{settings_name}.json"
        print(f"Running evaluation for {settings_name}:")

        print("Loading knowledge base embeddings...")
        knowledge_index = self.load_embeddings(
            self.load_fake_knowledge_base(),
            chunk_size=chunk_size,
        )

        print("Running RAG...")
        reranker = RAGPretrainedModel.from_pretrained(reranker_model)

        # Load synthetically generated eval_dataset from
        # https://huggingface.co/datasets/friedahuang/m-ric_huggingface_doc_347
        eval_dataset = datasets.load_dataset(eval_dataset, split="train")
        evaluation_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a fair evaluator language model."),
                HumanMessagePromptTemplate.from_template(
                    self.evaluation_prompt_template
                ),
            ]
        )

        eval_chat_model = ChatOllama(
            model=llm_model,
            temperature=0,
        )
        evaluator_name = "Llama3.1"

        self.run_rag_tests(
            answer_with_rag=self.answer_with_rag,
            eval_dataset=eval_dataset,
            knowledge_index=knowledge_index,
            output_file=output_file_name,
            reranker=reranker,
            verbose=True,
            test_settings=settings_name,
        )

        print("Running evaluation...")
        __class__.evaluate_answers(
            output_file_name,
            eval_chat_model,
            evaluator_name,
            evaluation_prompt_template,
        )

    @staticmethod
    def inspect_results():
        import glob

        outputs = []
        for file in glob.glob("./eval_output/*.json"):
            output = pd.DataFrame(json.load(open(file, "r")))
            output["settings"] = file
            outputs.append(output)
        result = pd.concat(outputs)

        result["eval_score_Llama3.1"] = result["eval_score_Llama3.1"].apply(
            lambda x: int(x) if isinstance(x, str) else 1
        )
        # Normalized the score
        result["eval_score_Llama3.1"] = (result["eval_score_Llama3.1"] - 1) / 4
        result.to_csv("./eval_output/evaluation_results.csv", index=False)
        for index, row in result.iterrows():
            print(
                f"Row {index}: Question: {row['question']} - Score: {row['eval_score_Llama3.1']}"
            )


# *Usage: Generate synthetic datasets
# e = QAGenerationEvaluator()
# e.save_synthetic_test_data()

# e.run_evals()
# e.inspect_results()
