def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    tokens = list(text.encode("utf-8"))

    for left, right , new_id in merges:

        merged = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == left
                and tokens[i + 1] == right
            ):
                merged.append(new_id)
                i += 2

            else:
                merged.append(tokens[i])
                i += 1

        tokens = merged
            
    return tokens

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    byte_values = bytearray()

    for token_ids in ids:
        if token_ids in vocab:
            byte_values.extend(vocab[token_ids])

        else:
            byte_values.append(token_ids)

    return byte_values.decode("utf-8")
