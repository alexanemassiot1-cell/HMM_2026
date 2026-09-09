from etat_transission import (
    forward_algorithm,
    backward_algorithm,
    transition_probabilities,
)

import numpy as np


# ensuite on met à jour la matrice de transision en fonction de ce qu'on appris précédemment
def update_transition_matrix(xi_list):

    n_states = xi_list[0].shape[1]

    # On additionne les probabilités de transition
    # de toutes les régions utilisées pour l'apprentissage
    total_xi = np.zeros(
        (n_states, n_states)
    )

    # On parcourt toutes les régions
    for xi in xi_list:

        for previous_state in range(n_states):

            for next_state in range(n_states):

                total_xi[
                    previous_state,
                    next_state
                ] += np.sum(
                    xi[
                        :,
                        previous_state,
                        next_state
                    ]
                )

    new_transition_matrix = np.zeros(
        (n_states, n_states)
    )

    # On normalise les probabilités de transition
    for previous_state in range(n_states):

        denominator = np.sum(
            total_xi[
                previous_state,
                :
            ]
        )

        if denominator > 0:

            for next_state in range(n_states):

                new_transition_matrix[
                    previous_state,
                    next_state
                ] = (
                    total_xi[
                        previous_state,
                        next_state
                    ] / denominator
                )

    return new_transition_matrix


# une fois qu'on a fait ca grace a baum welch on va donc répété jusqu'a ce que la matrice ne change plus
def baum_welch(
    emissions_regions,
    transition_matrix,
    initial_probabilities,
    max_iterations=100,
    tolerance=1e-6
):

    for iteration in range(max_iterations):  # répétitions de l'apprentissage plusieurs fois

        # Liste qui contient les transitions
        # de chaque région
        xi_list = []

        # On traite chaque région séparément
        for emissions in emissions_regions:

            # 1. Forward avant chaque bins
            forward, scaling = forward_algorithm(
                emissions,
                transition_matrix,
                initial_probabilities
            )

            # 2. Backward après chaque bins
            backward = backward_algorithm(
                emissions,
                transition_matrix,
                scaling
            )

            # 3. Probabilités des transitions :
            # Quelle est la probabilité que le modèle soit passé de α0 → α1, α1 → α1, etc. ?
            xi = transition_probabilities(
                forward,
                backward,
                emissions,
                transition_matrix
            )

            # On ajoute les transitions de cette région
            # à la liste des transitions
            xi_list.append(xi)

        # 4. Nouvelle matrice de transition
        # en utilisant toutes les régions
        new_transition_matrix = update_transition_matrix(
            xi_list
        )

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
        if difference < tolerance:  # si presque la matrice est la meme le modèle a convergé

            print(
                f"Baum-Welch convergé après "
                f"{iteration + 1} itérations"
            )

            break

    return transition_matrix