import os
import json
try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' python package is not installed. Run 'pip install openai'")
    raise

class OpenAIBackend:
    """Handles secure communication with OpenAI's API."""
    def __init__(self):
        # Automatically picks up OPENAI_API_KEY from environment variables
        self.client = OpenAI()
        
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Forces the LLM to output perfectly structured valid JSON."""
        response = self.client.chat.completions.create(
            model="gpt-4o", # Using a highly capable model for semantic reasoning
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Zero creativity, we want deterministic math
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
