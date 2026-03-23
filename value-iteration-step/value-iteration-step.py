def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    new_values = []
    for s in range(len(values)):
        q_values = []
        for a in range(len(rewards[s])):
            future = sum(transitions[s][a][s_prime] * values[s_prime] for s_prime in range(len(values)))
            q_values.append(rewards[s][a] + gamma * future)
        new_values.append(max(q_values))
    return new_values