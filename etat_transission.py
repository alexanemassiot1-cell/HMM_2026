import numpy as np
n_states = 3


transition_matrix = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90]
])

initial_probabilities = np.array([
    1.0,  # α0
    0.0,  # α1
    0.0   # α2
])

def transition_probability(previous_state, current_state, transition_matrix):
    return transition_matrix[previous_state][current_state]

transition_probability(0, 1, transition_matrix)

import numpy as np


def forward_algorithm(emissions, transition_matrix, initial_probabilities):
    """
    Algorithme Forward avec normalisation à chaque bin.

    emissions :
        matrice (nombre_de_bins, 3)
        colonnes = alpha0, alpha1, alpha2

    transition_matrix :
        matrice 3 x 3

    initial_probabilities :
        probabilités initiales des 3 états
    """

    n_bins = len(emissions)
    n_states = emissions.shape[1]

    forward = np.zeros((n_bins, n_states))
    scaling = np.zeros(n_bins)

    # Initialisation
    forward[0, :] = (
        initial_probabilities
        * emissions[0, :]
    )

    scaling[0] = np.sum(forward[0, :])

    if scaling[0] > 0:
        forward[0, :] /= scaling[0]

    # Récursion
    for i in range(1, n_bins):

        for current_state in range(n_states):

            total = 0.0

            for previous_state in range(n_states):

                total += (
                    forward[i - 1, previous_state]
                    * transition_matrix[
                        previous_state,
                        current_state
                    ]
                )

            forward[i, current_state] = (
                total * emissions[i, current_state]
            )

        # Normalisation
        scaling[i] = np.sum(forward[i, :])

        if scaling[i] > 0:
            forward[i, :] /= scaling[i]

    return forward, scaling


def backward_algorithm(
    emissions,
    transition_matrix,
    scaling
):
    """
    Algorithme Backward avec les mêmes facteurs
    de normalisation que Forward.
    """

    n_bins = len(emissions)
    n_states = emissions.shape[1]

    backward = np.zeros((n_bins, n_states))

    # Dernier bin
    backward[n_bins - 1, :] = 1.0

    # Récursion inverse
    for i in range(n_bins - 2, -1, -1):

        for current_state in range(n_states):

            total = 0.0

            for next_state in range(n_states):

                total += (
                    transition_matrix[
                        current_state,
                        next_state
                    ]
                    * emissions[i + 1, next_state]
                    * backward[i + 1, next_state]
                )

            backward[i, current_state] = total

        # Même normalisation que Forward
        if scaling[i + 1] > 0:
            backward[i, :] /= scaling[i + 1]

    return backward


def state_probabilities(forward, backward):
    """
    Calcule P(S_i = état | données)
    """

    n_bins = forward.shape[0]
    n_states = forward.shape[1]

    probabilities = np.zeros(
        (n_bins, n_states)
    )

    for i in range(n_bins):

        probabilities[i, :] = (
            forward[i, :]
            * backward[i, :]
        )

        total = np.sum(probabilities[i, :])

        if total > 0:
            probabilities[i, :] /= total

    return probabilities


def transition_probabilities(
    forward,
    backward,
    emissions,
    transition_matrix
):
    """
    Calcule xi(i,j) :
    probabilité d'être dans l'état i
    puis dans l'état j au bin suivant.
    """

    n_bins = forward.shape[0]
    n_states = forward.shape[1]

    xi = np.zeros(
        (n_bins - 1, n_states, n_states)
    )

    for i in range(n_bins - 1):

        total = 0.0

        for previous_state in range(n_states):

            for next_state in range(n_states):

                xi[i,
                   previous_state,
                   next_state] = (
                    forward[
                        i,
                        previous_state
                    ]
                    * transition_matrix[
                        previous_state,
                        next_state
                    ]
                    * emissions[
                        i + 1,
                        next_state
                    ]
                    * backward[
                        i + 1,
                        next_state
                    ]
                )

                total += xi[
                    i,
                    previous_state,
                    next_state
                ]

        if total > 0:
            xi[i, :, :] /= total

    return xi

# ensuite on met à jour la matrice de transision en focntion de ce qu'on appris précédemment 
def update_transition_matrix(xi):

    n_states = xi.shape[1]

    new_transition_matrix = np.zeros(
        (n_states, n_states)
    )

    for previous_state in range(n_states):

        denominator = np.sum(
            xi[:, previous_state, :]
        )

        if denominator > 0:

            for next_state in range(n_states):

                numerator = np.sum(
                    xi[
                        :,
                        previous_state,
                        next_state
                    ]
                )

                new_transition_matrix[
                    previous_state,
                    next_state
                ] = numerator / denominator

    return new_transition_matrix

# une fois qu'on a fait ca grace a baum welch on va donc répété jusqu'a ce que la matrice ne change plus 

