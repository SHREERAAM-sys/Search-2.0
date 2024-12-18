import json
from enum import Enum
from typing import List

import requests

COLPALI_BASE_URL = "http://54.188.147.87:8000/api/v1"
CODE_SEARCH_BASE_URL = "http://127.0.0.1:8000/api/v1"

class ToolNames(str, Enum):
    colpali_search = "colpali_search"
    colpali_embed = "colpali_embed"
    colpali_delete_file = "colpali_delete_file"
    colpali_get_all_files = "colpali_get_all_files"
    code_search_embed = "code_search_embed"
    code_search = "code_search"


def code_search_embed(filepaths: List[str]):
    #print("inside code search embed tool")
    try:
        files = [
            ("files", (filepath, open(filepath, "rb"), "text/plain"))
            for filepath in filepaths
        ]
        response = requests.post(f"{CODE_SEARCH_BASE_URL}/upload/files", files=files)

        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"

def code_search(query: str):
    try:
        data = {"query": query}
        response = requests.post(f"{CODE_SEARCH_BASE_URL}/search", data=data)
        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"



def colpali_search(query: str, top_k: int, email: str) -> dict:
    try:
        json_body = {"query": query, "top_k": top_k, "email": email}
        response = requests.post(f"{COLPALI_BASE_URL}/search", json=json_body)
        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"


def colpali_embed(filepaths: List[str]):
    try:
        files = [
            ("files", (filepath, open(filepath, "rb"), "application/pdf"))
            for filepath in filepaths
        ]
        response = requests.post(f"{COLPALI_BASE_URL}/embeddings/files", files=files)
        result = response.json()
        return json.dumps(result["metadata"])
    except Exception as e:
        return f"Error: {e}"


def colpali_delete_file(id: int):
    try:
        response = requests.delete(f"{COLPALI_BASE_URL}/files/{id}")
        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"


def colpali_get_all_files():
    try:
        response = requests.get(f"{COLPALI_BASE_URL}/files/")
        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"


tools = [
    {
        "name": "colpali_search",
        "description": "Performs a search based on a query and additional parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query must not be empty",
                },
                "top_k": {
                    "type": "integer",
                    "description": "top_k must be greater than 0",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the user making the request",
                },
            },
            "required": ["query", "top_k", "email"],
        },
    },
    {
        "name": "colpali_embed",
        "description": "Generates embeddings for PDF files",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of file paths to the PDF files",
                }
            },
            "required": ["filepaths"],
        },
    },
    {
        "name": "colpali_delete_file",
        "description": "Delete embedded files from the system",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The id of the file that the user intends to delete",
                }
            },
            "required": ["id"],
        },
    },
    {
        "name": "colpali_get_all_files",
        "description": "Get all embedded files",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "cache_control": {"type": "ephemeral"},
    },
    {
        "name": "code_search",
        "description": "Performs a search based on a query and retrieves relevant code snippets",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query; it must not be empty",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "code_search_embed",
        "description": "Generates embeddings for text-based code files",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of file paths to the code files",
                }
            },
            "required": ["filepaths"],
        },
    },
]
