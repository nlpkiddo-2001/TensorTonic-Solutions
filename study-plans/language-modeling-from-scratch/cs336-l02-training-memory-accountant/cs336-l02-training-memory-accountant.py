def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    parameters_count = 0
    gradients_count = 0
    activation_count = 0
    optimizer_count = 0
    total_count = 0


    def count_elements(shape):

        count = 1

        for dim in shape:
            count *= dim

        return count


    total_params = 0

    for shape in param_shapes:
        total_params += count_elements(shape)

    parameters_count += total_params * param_bytes_per_element
    gradients_count += total_params * grad_bytes_per_element

        
    total_activations = 0

    for shape in activation_shapes:
        total_activations += count_elements(shape)
        
    activation_count += total_activations * activation_bytes_per_element

    optimizer_state = 0
    
    if optimizer == "adagrad":
        optimizer_state = 1

    elif optimizer == "adam":
        optimizer_state = 2

    
        
    optimizer_count += total_params * optimizer_state * optimizer_bytes_per_element

    total_count += parameters_count + gradients_count + activation_count + optimizer_count

    return {
        "parameters" : parameters_count,
        "gradients" : gradients_count,
        "activations" : activation_count,
        "optimizer_state" : optimizer_count,
        "total" : total_count
    }
        

    

    
    
