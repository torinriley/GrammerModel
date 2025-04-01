import re
import nltk
from nltk.corpus import words
from difflib import get_close_matches, SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import iterative_correct
from correctnessPipeline import correctness_pipeline
from grammerPipeline import apply_grammar_pipeline

if __name__ == "__main__":
    while True:
        try:
            sentence = input("\n[Grammar Mode] Enter a sentence (or 'exit' to quit):\n> ")
            if sentence.strip().lower() == "exit":
                break
            result = apply_grammar_pipeline(sentence)
            print("Corrected:", result)
        except KeyboardInterrupt:
            break