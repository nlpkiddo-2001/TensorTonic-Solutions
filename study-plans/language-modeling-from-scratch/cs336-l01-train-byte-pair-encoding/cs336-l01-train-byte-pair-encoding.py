from collections import Counter

def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    token_bytes = {i : bytes([i]) for i in range(256)}

    sequences = [
        list(text.encode("utf-8"))
        for text in corpus
    ]

    vocab = []
    merges = []
    next_id = 256

    while next_id < vocab_size:

        # step 1 : count adjacent pairs
        pair_counts = Counter()

        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += 1


        if not pair_counts:
            break 

        # step 2 : find high freq
        best_pair = max(
            pair_counts,
            key = lambda x : (
                pair_counts[x],
                token_bytes[x[0]],
                token_bytes[x[1]]
            )
        )

        left, right = best_pair
        
        # step 3: create a new token
        new_id = next_id
        new_bytes = token_bytes[left] + token_bytes[right]

        token_bytes[new_id] = new_bytes
        vocab.append([new_id, list(new_bytes)])
        merges.append([left, right, new_id])

        # step 4: replace pairs from left to right
        for idx, seq in enumerate(sequences):
            new_seq = []
            i = 0

            while i < len(seq):

                if (
                    i + 1 < len(seq) 
                    and seq[i] == left
                    and seq[i+1] == right
                ):
                    new_seq.append(new_id)
                    i += 2

                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[idx] = new_seq


        next_id += 1

    return {
        "vocab" : vocab,
        "merges" : merges 
    }
        
        
                    
        
                
