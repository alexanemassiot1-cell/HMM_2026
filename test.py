import numpy as np

from baum_welch import baum_welch


# Émissions artificielles
# colonnes = alpha0, alpha1, alpha2
emissions = np.array([
    [0.90, 0.05, 0.05],
    [0.85, 0.10, 0.05],
    [0.80, 0.15, 0.05],

    [0.05, 0.90, 0.05],
    [0.05, 0.85, 0.10],
    [0.10, 0.80, 0.10],

    [0.05, 0.10, 0.85],
    [0.05, 0.10, 0.90],
    [0.10, 0.10, 0.80],

    [0.80, 0.10, 0.10],
    [0.85, 0.10, 0.05],
    [0.90, 0.05, 0.05]
])


# Matrice de transition initiale
transition_matrix = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90]
])


# L'article fixe l'état initial à alpha0
initial_probabilities = np.array([
    1.0,
    0.0,
    0.0
])


print("Matrice de transition initiale :")
print(transition_matrix)

print("\nDébut Baum-Welch...")

learned_transition_matrix = baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities
)

print("\nMatrice de transition apprise :")
print(learned_transition_matrix)