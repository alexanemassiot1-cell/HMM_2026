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

def forward_algorithm(emissions, transition_matrix, initial_probabilities):

    n_bins = len(emissions)
    n_states = 3

    forward = np.zeros((n_bins, n_states))

    # Premier bin
    for state in range(n_states):
        forward[0, state] = (
            initial_probabilities[state]
            * emissions[0, state]
        )

    # Bins suivants
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
                total
                * emissions[i, current_state]
            )

    return forward


def backward_algorithm(emissions, transition_matrix):

    n_bins = len(emissions)
    n_states = 3

    backward = np.zeros((n_bins, n_states)) 

    # Dernier bin
    for state in range(n_states):
        backward[n_bins - 1, state] = 1.0 # Pour le dernier bin, on met 1.0 parce qu'il n'y a plus de bin après lui.

    # On remonte vers le premier bin
    for i in range(n_bins - 2, -1, -1): # On commence à l'avant-dernier bin et on remonte jusqu'au premier.

        for current_state in range(n_states):

            total = 0.0

            for next_state in range(n_states):

                total += (
                    transition_matrix[ #quelle est la probabilité de passer de l'état actuel à l'état du bin suivant ?
                        current_state,
                        next_state
                    ]
                    * emissions[i + 1, next_state] #correspond à la probabilité d'observer les données du bin suivant si celui-ci est dans next_state.
                    * backward[i + 1, next_state] #contient déjà les informations provenant de tous les bins situés après.
                )

            backward[i, current_state] = total

    return backward

def state_probabilities(forward, backward):

    n_bins = forward.shape[0]
    n_states = forward.shape[1]

    probabilities = np.zeros((n_bins, n_states))

    for i in range(n_bins):

        total = 0.0

        # On combine Forward et Backward
        for state in range(n_states):
            probabilities[i, state] = (
                forward[i, state]
                * backward[i, state]
            )
            total += probabilities[i, state]

        # Normalisation
        if total > 0:
            probabilities[i, :] /= total

    return probabilities

# Partie Baum welch qui apprend la matrice de transision 
"""
xi apprend la probabilité de chaque transision
et backward et fordward permet de voir ce qui se passe avant et après pour observer la probalité de maintenant donc xi prend tout en considération
"""
def transition_probabilities(forward, backward, emissions, transition_matrix):
    
    n_bins = forward.shape[0]
    n_states = forward.shape[1]

    xi = np.zeros((n_bins - 1, n_states, n_states))

    for i in range(n_bins - 1):

        total = 0.0

        for previous_state in range(n_states):

            for next_state in range(n_states):

                xi[i, previous_state, next_state] = (
                    forward[i, previous_state]
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

        # Normalisation
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

def baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities,
    max_iterations=100,
    tolerance=1e-6
):

    for iteration in range(max_iterations): # répétitions de l'apprentissage plusieurs fois 

        # 1. Forward avant chaque bins
        forward = forward_algorithm(
            emissions,
            transition_matrix,
            initial_probabilities
        )

        # 2. Backward après chaque bins 
        backward = backward_algorithm(
            emissions,
            transition_matrix
        )

        # 3. Probabilités des transitions : Quelle est la probabilité que le modèle soit passé de α0 → α1, α1 → α1, etc. ?
        xi = transition_probabilities(
            forward,
            backward,
            emissions,
            transition_matrix
        )

        # 4. Nouvelle matrice de transition
        new_transition_matrix = update_transition_matrix(xi)

        # 5. Vérifier si la matrice a suffisamment peu changé
        difference = np.max(
            np.abs(
                new_transition_matrix
                - transition_matrix
            )
        )

        # 6. Mettre à jour la matrice
        transition_matrix = new_transition_matrix

        # 7. Arrêt si le modèle est stabilisé (on compare l'ancienne et la nouvelle)
        if difference < tolerance: # si presque la matrice est la meme le modèle a convergé 
            print(
                f"Baum-Welch convergé après "
                f"{iteration + 1} itérations"
            )
            break

    return transition_matrix

#test 
emissions = np.array([
    [0.90, 0.05, 0.05],  # α0
    [0.85, 0.10, 0.05],  # α0
    [0.80, 0.15, 0.05],  # α0

    [0.05, 0.90, 0.05],  # α1
    [0.05, 0.85, 0.10],  # α1
    [0.10, 0.80, 0.10],  # α1

    [0.05, 0.10, 0.85],  # α2
    [0.05, 0.10, 0.90],  # α2
    [0.10, 0.10, 0.80],  # α2

    [0.80, 0.10, 0.10],  # α0
    [0.85, 0.10, 0.05],  # α0
    [0.90, 0.05, 0.05]   # α0
])

learned_transition_matrix = baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities
)

print(learned_transition_matrix)