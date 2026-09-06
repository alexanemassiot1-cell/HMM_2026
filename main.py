from count_reads import count_reads
from posterior import posterior_parameters, posterior_mean
from emission import calculate_emissions
from baum_welch import baum_welch


import numpy as np

from baum_welch import baum_welch


es_counts = count_reads(
    "GSM307619_ES.H3K27me3.aligned.txt.gz"
)

np_counts = count_reads(
    "GSM307614_NP.H3K27me3.aligned.txt.gz"
)

# Paramètres du posterior
a1, b1 = posterior_parameters(x_es, n1, m)
a2, b2 = posterior_parameters(x_np, n2, m)

# Moyenne du posterior
p_es = posterior_mean(a1, b1)
p_np = posterior_mean(a2, b2)


# Émissions artificielles
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


# État initial : α0
initial_probabilities = np.array([
    1.0,
    0.0,
    0.0
])


# Baum-Welch
learned_transition_matrix = baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities
)

print(learned_transition_matrix)
emissions = calculate_emissions(
    bins,
    n1,
    n2,
    m,
    tau
)