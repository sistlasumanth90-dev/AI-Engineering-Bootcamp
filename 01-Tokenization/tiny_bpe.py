from collections import Counter
from typing import Dict, List, Tuple


def split_word(word: str) -> Tuple[str, ...]:
    """
    Split a word into individual characters.
    Example:
        "walk" -> ("w", "a", "l", "k")
    """
    return tuple(word)


def count_pairs(vocabulary: Dict[Tuple[str, ...], int]) -> Counter:
    """
    Count adjacent token pairs across all words.

    The dictionary value represents how many times
    that word appears in the training data.
    """
    pair_counts = Counter()

    for tokens, frequency in vocabulary.items():
        for index in range(len(tokens) - 1):
            pair = (tokens[index], tokens[index + 1])
            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(
    pair_to_merge: Tuple[str, str],
    vocabulary: Dict[Tuple[str, ...], int],
) -> Dict[Tuple[str, ...], int]:
    """
    Merge every occurrence of a selected adjacent pair.
    Example:
        ("w", "a") -> "wa"
    """
    merged_vocabulary = {}

    for tokens, frequency in vocabulary.items():
        new_tokens: List[str] = []
        index = 0

        while index < len(tokens):
            if (
                index < len(tokens) - 1
                and tokens[index] == pair_to_merge[0]
                and tokens[index + 1] == pair_to_merge[1]
            ):
                new_tokens.append(tokens[index] + tokens[index + 1])
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1

        merged_vocabulary[tuple(new_tokens)] = frequency

    return merged_vocabulary


def print_vocabulary(
    vocabulary: Dict[Tuple[str, ...], int],
) -> None:
    """
    Print the current tokenization of every word.
    """
    for tokens, frequency in vocabulary.items():
        print(f"{' '.join(tokens):20} frequency={frequency}")


def train_bpe(words: List[str], number_of_merges: int = 5) -> None:
    """
    Train a very small BPE tokenizer.

    This is intentionally simplified for learning.
    """
    vocabulary = Counter(tuple(split_word(word)) for word in words)

    print("Initial vocabulary")
    print("-" * 40)
    print_vocabulary(vocabulary)

    for merge_number in range(1, number_of_merges + 1):
        pair_counts = count_pairs(vocabulary)

        if not pair_counts:
            print("\nNo more pairs available.")
            break

        most_common_pair, frequency = pair_counts.most_common(1)[0]

        print(f"\nMerge {merge_number}")
        print("-" * 40)
        print("Most frequent pair:", most_common_pair)
        print("Frequency:", frequency)

        vocabulary = merge_pair(most_common_pair, vocabulary)

        print("\nVocabulary after merge:")
        print_vocabulary(vocabulary)

if __name__ == "__main__":
    training_words = [
        "play",
        "playing",
        "played",
        "player",
        "replaying",
    ]

    train_bpe(training_words, number_of_merges=6)

    train_bpe(training_words, number_of_merges=6)