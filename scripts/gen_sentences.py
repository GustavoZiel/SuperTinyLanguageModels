import argparse
import random

import pyperclip

from trainers.utils import DATASET_DICT, load_custom_dataset

# random.seed(42)


def load_data(
    dataset_name: str,
):
    if dataset_name not in DATASET_DICT:
        dataset = load_custom_dataset(dataset_name)
    else:
        dataset = DATASET_DICT[dataset_name]()
    return dataset


def select_random_sentence(sentences, min_len=None, max_len=None):
    """Select a random sentence from a list, optionally filtering by min and/or max length."""
    filtered = sentences
    if min_len is not None:
        filtered = [s for s in filtered if len(s.split()) >= min_len]
    if max_len is not None:
        filtered = [s for s in filtered if len(s.split()) <= max_len]
    if filtered:
        return random.choice(filtered)
    else:
        return None
        # print("No sentences found with length", end="")
        # if min_len is not None and max_len is not None:
        #     print(f" between {min_len} and {max_len}. Selecting from all sentences.")
        # elif min_len is not None:
        #     print(f" >= {min_len}. Selecting from all sentences.")
        # elif max_len is not None:
        #     print(f" <= {max_len}. Selecting from all sentences.")
        # else:
        #     print(". Selecting from all sentences.")
        # return random.choice(sentences)


def extract_sentences(text):
    """Split text into sentences using '. ' as a separator."""
    return text.split(". ")


def build_sentence_answer_dict(sentence, n=3):
    """Split a sentence into a prompt and an answer (last n words)."""
    words = sentence.split()
    if len(words) <= n:
        return {"sentence": "", "answer": " " + sentence}
    answer = " ".join(words[-n:])
    sentence_part = " ".join(words[:-n])
    return {"sentence": sentence_part, "answer": " " + answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (HuggingFace format, e.g. 'ag_news')",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--num_sentences", type=int, default=5, help="Number of sentences to sample"
    )
    parser.add_argument(
        "--min_context_len", type=int, default=10, help="Minimum number of words in the context"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=50, help="Maximum number of words in the context"
    )
    parser.add_argument("--answer_len", type=int, default=3, help="Number of words in the answer")
    args = parser.parse_args()

    df = load_data(args.dataset)
    data_split = df[args.split]

    num_sentences = args.num_sentences
    answer_len = args.answer_len
    sentence_answer_list = []
    i = 0
    while i < num_sentences:
        random_idx = random.randint(0, len(data_split) - 1)
        sample = data_split[random_idx]
        sample_title = sample.get("title", "").strip()
        sample_text = sample.get("text", "")

        sentences = extract_sentences(sample_text)
        if len(sentences) < answer_len:
            continue

        sentence = select_random_sentence(
            sentences, min_len=args.min_context_len, max_len=args.max_context_len
        )
        if not sentence:
            continue
        sentence_dict = build_sentence_answer_dict(sentence, n=answer_len)
        sentence_answer_list.append({"title": sample_title, **sentence_dict})

        i += 1

    s = ""
    for prompt in sentence_answer_list:
        s += f'- sentence: "{prompt["sentence"]}"\n  answer: "{prompt["answer"]}"\n  title: "{prompt["title"]}"\n'
    pyperclip.copy(s)
    print(s)


if __name__ == "__main__":
    main()
