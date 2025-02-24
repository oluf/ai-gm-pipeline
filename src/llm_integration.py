import subprocess
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for Llama.cpp
LLAMA_BINARY = "/home/oluf/bin/llama.cpp/build/bin/llama-run"
MODEL_FILE = "/home/oluf/projects/ai-gm-pipeline/models/mythomax-13B.Q4_K_M.gguf"

GPU_LAYERS = "35"
TEMPERATURE = "0.7"
TIMEOUT = 30


class LLMIntegration:
    def __init__(self):
        logging.info("LLM Integration initialized.")

    @staticmethod
    def clean_ansi(text: str) -> str:
        """Removes ANSI escape sequences from output."""
        return re.sub(r'\x1b\[[0-9;]*[mK]', '', text)

    @staticmethod
    def generate_response(query: str, context: str) -> str:
        """Send query + retrieved context to Llama.cpp and return the response."""
        full_prompt = f"Using the following rulebook context:\n\n{context}\n\nAnswer the player's question: {query}"

        command = [
            LLAMA_BINARY,
            "--ngl", GPU_LAYERS,
            "--temp", TEMPERATURE,
            MODEL_FILE,
            full_prompt  # Ensure the full prompt is a single string argument
        ]

        logging.info("Running Llama.cpp with command: %s", " ".join(command))

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                logging.error("Llama.cpp Error: %s", error.decode())

            result = LLMIntegration.clean_ansi(output.decode()).strip()

            logging.info("Llama Output: %s", result)
            return result
        except Exception as e:
            logging.exception("Exception occurred while running Llama.cpp")
            return "Error occurred during LLM processing."


if __name__ == "__main__":
    llm = LLMIntegration()