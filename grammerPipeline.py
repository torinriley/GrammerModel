import spacy
import re
import inflect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset
import json
from sentence_transformers import SentenceTransformer, util
import errant
import spacy as errant_spacy
from transformers import BertTokenizer, BertModel
import torch

with open("conjugation_map.json") as f:
    conjugation_map = json.load(f)

nlp = spacy.load("en_core_web_sm")

ml_model_name = "prithivida/grammar_error_correcter_v1"
ml_tokenizer = AutoTokenizer.from_pretrained(ml_model_name)
ml_model = AutoModelForSeq2SeqLM.from_pretrained(ml_model_name)
grammar_classifier = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_ungrammatical(text):
    try:
        prediction = grammar_classifier(text)[0]
        return prediction["label"] == "LABEL_0"  # CoLA: LABEL_0 = ungrammatical
    except:
        return False

def detect_tense(doc):
    for token in doc:
        if token.pos_ == "VERB":
            if token.tag_ in {"VBD", "VBN"}:
                return "past"
            elif token.tag_ == "VBZ":
                return "present"
    return "unknown"

def subject_verb_agreement_rule(text, doc):
    changes = 0
    tense = detect_tense(doc)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text.lower()
            verb = token.head.text.lower()
            if verb in conjugation_map:
                if tense in conjugation_map[verb]:
                    conjugated = conjugation_map[verb][tense]
                elif subject in conjugation_map[verb]:
                    conjugated = conjugation_map[verb][subject]
                else:
                    continue
                new_text = re.sub(rf"\b{token.text}\s+{verb}\b", f"{token.text} {conjugated}", text, flags=re.IGNORECASE)
                if new_text != text:
                    changes += 1
                    text = new_text
    return text, changes

def plural_was_were_rule(text):
    new_text = re.sub(r'\b(they|we|you)\s+was\b', lambda m: f"{m.group(1)} were", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def i_has_rule(text):
    new_text = re.sub(r'\bI has\b', 'I have', text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def third_person_have_rule(text):
    new_text = re.sub(r'\b(he|she|it) have\b', lambda m: f"{m.group(1)} has", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def dont_doesnt_rule(text):
    new_text = re.sub(r'\b(he|she|it) don\'t\b', lambda m: f"{m.group(1)} doesn\'t", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def article_before_vowel_rule(text, doc):
    changes = 0
    for chunk in doc.noun_chunks:
        if chunk.start > 0:
            prev = doc[chunk.start - 1]
            if prev.text.lower() == "a" and chunk[0].text[0].lower() in "aeiou":
                new_text = re.sub(rf"\b{prev.text} {chunk[0].text}\b", f"{prev.text}n {chunk[0].text}", text)
                if new_text != text:
                    changes += 1
                    text = new_text
    return text, changes

def their_theyre_rule(text, doc):
    changes = 0
    for token in doc:
        if token.tag_ == "VBG" and token.i > 0:
            prev_token = doc[token.i - 1]
            if prev_token.text.lower() == "their":
                new_text = re.sub(r'\btheir (\w+ing)\b', r"they're \1", text, flags=re.IGNORECASE)
                if new_text != text:
                    changes += 1
                    text = new_text
    return text, changes

def article_an_rule(text):
    # Fix "a" to "an" for cases like "a apple", "a owl", "a honor"
    changes = 0
    new_text = re.sub(r'\b([Aa]) ([aeiouhAEIOUH])', r'\1n \2', text)
    if new_text != text:
        changes += 1
        text = new_text
    return text, changes

def their_theyre_possessive_rule(text, doc):
    # Improve "their" correction before present participle or verb
    changes = 0
    for token in doc:
        if token.tag_ in {"VBG", "VBZ"} and token.i > 0:
            prev_token = doc[token.i - 1]
            if prev_token.text.lower() == "their":
                new_text = re.sub(r'\btheir (\w+)', r"they're \1", text, flags=re.IGNORECASE)
                if new_text != text:
                    changes += 1
                    text = new_text
    return text, changes

def i_is_am_rule(text):
    new_text = re.sub(r'\bI is\b', 'I am', text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def plural_subject_is_are_rule(text):
    new_text = re.sub(r'\b(they|we|you) is\b', lambda m: f"{m.group(1)} are", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def correct_went_vs_gone_rule(text):
    new_text = re.sub(r'\b(has|have)\s+went\b', lambda m: f"{m.group(1)} gone", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def do_does_rule(text):
    new_text = re.sub(r'\b(it|he|she)\s+do not\b', lambda m: f"{m.group(1)} doesn\'t", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def their_there_was_rule(text):
    new_text = re.sub(r'\bTheir was\b', 'There were', text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def add_be_verb_rule(text):
    new_text = re.sub(r'\b(This|That|It) an?\b', lambda m: f"{m.group(1)} is an", text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def base_verb_s_rule(text):
    verb_map = {"say": "says", "know": "knows", "eat": "eats", "chase": "chases"}
    changes = 0
    for subject in ["he", "she", "it"]:
        for base, correct in verb_map.items():
            pattern = rf'\b{subject}\s+{base}\b'
            new_text = re.sub(pattern, f"{subject} {correct}", text, flags=re.IGNORECASE)
            if new_text != text:
                changes += 1
                text = new_text
    return text, changes

def modal_should_have_rule(text):
    new_text = re.sub(r'\b(could|would|should|might|must|may) of\b', r'\1 have', text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def fix_have_went_after_modal(text):
    new_text = re.sub(r'\bhave went\b', 'have gone', text, flags=re.IGNORECASE)
    return new_text, int(new_text != text)

def ml_grammar_corrector(text):
    input_text = f"gec: {text}"
    input_ids = ml_tokenizer.encode(input_text, return_tensors="pt")
    outputs = ml_model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    corrected = ml_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Define or load the context_tokenizer and context_model
context_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
context_model = BertModel.from_pretrained("bert-base-uncased")

def score_contextual_fit(original, candidate):
    inputs = context_tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = context_model(**inputs)
    return torch.mean(outputs.last_hidden_state).item()

def rank_by_similarity(original, rule_output, ml_output):
    sentences = [original, rule_output, ml_output]
    embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
    sim_rule = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    sim_ml = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()
    return rule_output if sim_rule > sim_ml else ml_output

def analyze_edits(source, corrected):
    annotator = errant.load("en")
    orig_doc = annotator.parse(source)
    cor_doc = annotator.parse(corrected)
    edits = annotator.annotate(orig_doc, cor_doc)
    return [str(edit) for edit in edits]

def simple_grammar_fixes(text):
    def apply_rules(text):
        score = 0
        doc = nlp(text)
        text, delta = subject_verb_agreement_rule(text, doc); score += delta
        text, delta = plural_was_were_rule(text); score += delta
        text, delta = i_has_rule(text); score += delta
        text, delta = third_person_have_rule(text); score += delta
        text, delta = dont_doesnt_rule(text); score += delta
        doc = nlp(text)
        text, delta = article_before_vowel_rule(text, doc); score += delta
        text, delta = article_an_rule(text); score += delta
        doc = nlp(text)
        text, delta = their_theyre_possessive_rule(text, doc); score += delta
        text, delta = i_is_am_rule(text); score += delta
        text, delta = plural_subject_is_are_rule(text); score += delta
        text, delta = correct_went_vs_gone_rule(text); score += delta
        text, delta = do_does_rule(text); score += delta
        text, delta = their_there_was_rule(text); score += delta
        text, delta = add_be_verb_rule(text); score += delta
        text, delta = base_verb_s_rule(text); score += delta
        text, delta = modal_should_have_rule(text); score += delta
        text, delta = fix_have_went_after_modal(text); score += delta
        return text, score

    # Pass 1: Rules
    corrected_text, score_1 = apply_rules(text)

    # Pass 2: ML if needed
    if is_ungrammatical(corrected_text):
        corrected_text = ml_grammar_corrector(corrected_text)

    # Pass 3: Reapply rules after ML
    corrected_text, score_2 = apply_rules(corrected_text)

    # Pass 4: Final ML sweep if still ungrammatical
    if is_ungrammatical(corrected_text):
        corrected_text = ml_grammar_corrector(corrected_text)

    # Rank outputs
    ml_output = ml_grammar_corrector(corrected_text)
    rule_score = score_contextual_fit(text, corrected_text)
    ml_score = score_contextual_fit(text, ml_output)
    final_output = corrected_text if rule_score > ml_score else ml_output
    return final_output

def correct_sentence(sentence: str) -> str:
    return simple_grammar_fixes(sentence)

def apply_grammar_pipeline(text: str) -> str:
    return correct_sentence(text)

if __name__ == "__main__":
    print("Grammar Corrector CLI. Type a sentence and press Enter (type 'exit' to quit):\n")
    while True:
        try:
            user_input = input("Input: ")
            if user_input.strip().lower() == "exit":
                break
            corrected = apply_grammar_pipeline(user_input)
            print(f"Corrected: {corrected}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break