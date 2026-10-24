def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """

    def _helper(sub_block : list[int]):
        res = 1
        for num in sub_block:
            res *= num
        return res

    forward_flops = 0

    for mat in matmuls:
        curr_flops = _helper(mat)
        forward_flops += curr_flops

    forward_flops *= 2
    forward_flops += attention_flops
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops

    return {
        "forward_flops" : forward_flops,
        "backward_flops" : backward_flops,
        "total_flops" : total_flops
    }
