import re
import torch

re_sentence_splitter = re.compile('([.!?﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def split_sentence(sentence):
    s = sentence
    sent_list = []
    for i in re_sentence_splitter.split(s):
        if re_sentence_splitter.match(i) and sent_list:
            sent_list[-1] += i
        elif i:
            sent_list.append(i)
    return sent_list


def extract_last_sentences_from_thought(thought, num_last_several_sentences=3):
    sentences_init = split_sentence(thought)

    sentences = []
    for _sent in sentences_init:
        if _sent != "" and _sent != "\n":
            sentences.append(_sent)

    # Get the last two sentences
    last_several_sentences = sentences[-num_last_several_sentences:]

    return last_several_sentences


def extract_last_sentences_str_from_thought(thought, num_last_several_sentences=3):
    last_several_sentences = extract_last_sentences_from_thought(
        thought, num_last_several_sentences=num_last_several_sentences
    )
    return "".join(last_several_sentences)


def tokenize_without_special_tokens(tokenizer, text):
    """Use tokenizer to get tokenized ids (without special tokens)"""
    return tokenizer(text, add_special_tokens=False).input_ids
