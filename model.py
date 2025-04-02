import re
import nltk
from nltk.corpus import words
from difflib import get_close_matches, SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import iterative_correct
from correctnessPipeline import correctness_pipeline


if __name__ == "__main__":
    print("Spell & Grammar Correction CLI. Type a sentence and press Enter (type 'exit' to quit):\n")
    while True:
        try:
            user_input = input("Input: ")
            if user_input.strip().lower() == "exit":
                break
            corrected = iterative_correct(user_input)
            print(f"Corrected: {corrected}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break