import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set it in the .env file.")


class QASystem:
    def __init__(self):
        """
        Initializes the Q&A system with Hugging Face API.
        """
        self.api_url = "https://api-inference.huggingface.co/models/deepset/roberta-large-squad2"
        self.api_token = HF_API_TOKEN
        if not self.api_token:
            raise ValueError("Hugging Face API token not found.")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def answer_question(self, question: str, context: str) -> str:
        """
        Sends a request to the Hugging Face API to answer a question.

        Args:
            question (str): The question to answer.
            context (str): The context to search for the answer.

        Returns:
            str: The answer to the question.
        """
        payload = {
            "inputs": {"question": question, "context": context},
            "parameters": {"max_answer_len": 1000, "min_score": 0.7},  # Adjust parameters as needed
        }

        try:
            # Debugging: Log the payload and response
            print(f"Request Payload: {payload}")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            print(f"API Response: {response.text}")  # Debugging: Log the raw API response
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            return data.get("answer", "No answer found")
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error processing the question"
