# This is a script to read and test SQuAD original file style Japanese question.
import argparse
from os.path import isfile
import json

import re
import numpy as np
import tensorflow as tf

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, PreserveParagraphs
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable
from docqa.squad.squad_official_evaluation import OfficialEvaluator

"""
Script to run a model on user provided question/context document.
This demonstrates how to use our document-pipeline on new input
"""

tokenizer = NltkAndPunctTokenizer()

def read_squad_style_database(ja_filepath):
    ja_questions = json.load(open(ja_filepath, "r"))["data"]

    paragraphs = {}
    questions = []

    for item in ja_questions:
        title = item["title"]
        paragraphs[item["title"]] = [preprocess_paragraph(paragraph["context"]) for paragraph in item["paragraphs"]]
        para_idx = 0

        for paragraph in item["paragraphs"]:
            for qa in paragraph["qas"]:
                qa["para_idx"] = para_idx
                qa["title"] = title
                qa["question"] = preprocess_question(qa["question"])
                questions.append(qa)
            para_idx += 1
    return paragraphs, questions

def preprocess_question(question):
    question = tokenizer.tokenize_paragraph_flat(question)
    return question

def preprocess_paragraph(paragraph):
    paragraph = tokenizer.tokenize_paragraph(paragraph)
    return paragraph

def main():
    parser = argparse.ArgumentParser(description="Run an ELMo model on user input")
    parser.add_argument("model", help="Model directory")
    parser.add_argument("ja_filepath", help="File path to japanese questions")
    parser.add_argument("result_file", help="File path to predicted result json")
    args = parser.parse_args()
    print(args)

    print("Preprocessing...")

    paragraphs, questions = read_squad_style_database(args.ja_filepath)
    # Load the model
    model_dir = ModelDir(args.model)
    model = model_dir.get_model()
    if not isinstance(model, ParagraphQuestionModel):
        raise ValueError("This script is built to work for ParagraphQuestionModel models only")

    paragraphs, questions = read_squad_style_database(args.ja_filepath)
    predictions = {}
    predictions["conf"] = {}
    for qa in questions:
        print(qa["id"])

        title = qa["title"]
        para_idx = qa["para_idx"]

        context = paragraphs[title][para_idx]
        question = qa["question"]

        print(context)
        print(question)

        if model.preprocessor is not None:
            context = [model.preprocessor.encode_text(question, x) for x in context]

        print("Setting up model")

        voc = set(question)
        for txt in context:
            voc.update(txt)
        model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), voc)

        print("Build tf graph")
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with sess.as_default():
            best_spans, conf = model.get_prediction().get_best_span(8)

        # Loads the saved weights
        model_dir.restore_checkpoint(sess)

        data = [ParagraphAndQuestion(x, question, None, "user-question%d"%i)
                for i, x in enumerate(context)]

        print("Starting run")

        encoded = model.encode(data, is_train=False)  # batch of `ContextAndQuestion` -> feed_dict
        best_spans, conf = sess.run([best_spans, conf], feed_dict=encoded)  # feed_dict -> predictions
        print(best_spans)
        predictions[qa["id"]] = best_spans
        predictions["conf"][qa["id"]] = conf
        print(predictions)

    result_f = open(args.result_file, "w")
    json.dump(predictions,result_f)
    exit()
    official_evaluator = OfficialEvaluator(args.ja_filepath, args.result_file)
    evaluation = official_evaluator.evaluate()
    print(evaluation)

if __name__ == "__main__":
    main()
