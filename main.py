from count_reads import count_reads
from posterior import posterior_parameters, posterior_mean
from emission import calculate_emissions, emission_probability, build_emission_lookup, emissions_from_lookup
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
eta = 0.7


print("Nombre total de fragments ESC :", n1)
print("Nombre total de fragments NPC :", n2)
print("Nombre de bins :", m)

#Sélection bins candidtats
threshold = 2 / (m * eta)

candidate_bins = []

for chromosome, start, x1, x2 in bins:

    F = (x1 / n1) + (x2 / n2)

    if F > threshold:

        candidate_bins.append(
            (chromosome, start, x1, x2)
        )


print("Seuil F :", threshold)
print("Nombre de bins candidats :", len(candidate_bins))

unique_count_pairs = set()

for chromosome, start, x1, x2 in candidate_bins:
    unique_count_pairs.add((x1, x2))

print("Nombre de couples (xESC, xNPC) différents :",
      len(unique_count_pairs))

print("Construction de la lookup table...")

emission_lookup = build_emission_lookup(
    candidate_bins,
    n1,
    n2,
    m,
    tau
)

emissions = emissions_from_lookup(
    candidate_bins,
    emission_lookup
)

print("Matrice des émissions :", emissions.shape)


#pour pas faire sur toute l'echantilions on fait sur 1000 aléatoirement
test_candidate_bins = candidate_bins[:1000]

test_emissions = emissions_from_lookup(
    test_candidate_bins,
    emission_lookup
)

print("Test emissions :", test_emissions.shape)
# 4. Initialisation de la matrice de transition
# TEST BAUM-WELCH SUR 1000 BINS RÉELS

test_emissions = emissions[:1000]

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

print("Matrice test :", test_emissions.shape)
print("Début Baum-Welch sur 1000 bins réels...")

learned_transition_matrix = baum_welch(
    test_emissions,
    transition_matrix,
    initial_probabilities
)

print("Matrice de transition apprise :")
print(learned_transition_matrix)

"""
print("Début de Baum-Welch sur les données réelles...")

# 5. Apprentissage de la matrice de transition
learned_transition_matrix = baum_welch(
    emissions,
    transition_matrix,
    initial_probabilities
)

print("Matrice de transition apprise :")
print(learned_transition_matrix)


#_---------------------------
# Test sur seulement 1 bins
print("TEST EMISSION")
print("n1 =", n1)
print("n2 =", n2)
print("m =", m)

x1 = 30
x2 = 2

a1, b1 = posterior_parameters(x1, n1, m)
a2, b2 = posterior_parameters(x2, n2, m)

print("a1,b1 =", a1, b1)
print("a2,b2 =", a2, b2)

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
"""
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