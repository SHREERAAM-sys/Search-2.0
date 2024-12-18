from transformers import pipeline

class RAGProcessor:
    _instance = None

    def __new__(cls, model_name: str = "EleutherAI/gpt-neo-1.3B"):
        # If no instance exists, create one
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.chat_model = pipeline("text-generation", model=model_name)
        return cls._instance

    def generate_rag_response(self, query: str, context: str) -> str:
        # Combine the query and context
        prompt = (
            f"You are a helpful assistant.\nUser Query: {query}\n\nContext:\n{context}\n\nResponse:"
        )

        # Generate the response
        response = self.chat_model(prompt, max_length=200, num_return_sequences=1)
        return response[0]["generated_text"]
