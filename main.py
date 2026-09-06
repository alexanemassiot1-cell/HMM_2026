from count_reads import count_reads
from posterior import posterior_parameters, posterior_mean
from emission import calculate_emissions, emission_probability
from baum_welch import baum_welch


import numpy as np


# 1. Lecture et comptage
es_counts = count_reads("Files/GSM307619_ES.H3K27me3.aligned.txt.gz")
np_counts = count_reads("Files/GSM307614_NP.H3K27me3.aligned.txt.gz")

# 2. Création des bins
all_bins = set(es_counts) | set(np_counts)

bins = []

for chromosome, start in sorted(all_bins):

    x1 = es_counts.get((chromosome, start), 0)
    x2 = np_counts.get((chromosome, start), 0)

    bins.append(
        (chromosome, start, x1, x2)
    )

# 3. Paramètres globaux
n1 = sum(es_counts.values())
n2 = sum(np_counts.values())

m = len(all_bins)
tau = 3.0

# Test sur seulement 10 bins
test_bins = [
    ("chr1", 0, 30, 2),
    ("chr1", 1000, 2, 30),
    ("chr1", 2000, 10, 10)
]

a1, b1 = posterior_parameters(30, n1, m)
a2, b2 = posterior_parameters(2, n2, m)

e0 = emission_probability(
    30, 2,
    n1, n2,
    a1, b1,
    a2, b2,
    m, tau,
    0
)

e1 = emission_probability(
    30, 2,
    n1, n2,
    a1, b1,
    a2, b2,
    m, tau,
    1
)

e2 = emission_probability(
    30, 2,
    n1, n2,
    a1, b1,
    a2, b2,
    m, tau,
    2
)

print("Emission α0 :", e0)
print("Emission α1 :", e1)
print("Emission α2 :", e2)

"""emissions = calculate_emissions(
    test_bins,
    n1,
    n2,
    m,
    tau
)

print(emissions)

# 4. Calcul des émissions
emissions = calculate_emissions(
    bins,
    n1,
    n2,
    m,
    tau
)
transition_matrix = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90]
])

initial_probabilities = np.array([
    1.0,
    0.0,
    0.0
])


# 5. HMM / Baum-Welch
learned_transition_matrix = baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities
)

# 6. Sauvegarde du résultat
with open("resultats_HMM.tsv", "w") as out:

    out.write("chrom\tstart\tES\tNP\talpha0\talpha1\talpha2\n")

    for i, (chromosome, start, x1, x2) in enumerate(bins):

        out.write(
            f"{chromosome}\t"
            f"{start}\t"
            f"{x1}\t"
            f"{x2}\t"
            f"{emissions[i, 0]}\t"
            f"{emissions[i, 1]}\t"
            f"{emissions[i, 2]}\n"
        )



print(learned_transition_matrix)
emissions = calculate_emissions(
    bins,
    n1,
    n2,
    m,
    tau
)
"""