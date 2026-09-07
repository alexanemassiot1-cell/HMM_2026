from count_reads import count_reads
from posterior import posterior_parameters, posterior_mean
from emission import calculate_emissions, emission_probability, build_emission_lookup, emissions_from_lookup
from baum_welch import baum_welch
from etat_transission import (
    forward_algorithm,
    backward_algorithm,
    state_probabilities
)

import numpy as np
import random

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


# 4. Sélection aléatoire de 10 000 bins pour l'apprentissage

random.seed(42)

n_training = 10000

training_indices = random.sample(
    range(len(candidate_bins)),
    n_training
)
training_indices.sort() # afin d'éviter que ce soit vraiment aléatoire et plus en adéquation avec le génome
training_emissions = emissions[training_indices]

print(
    "Nombre de régions utilisées pour l'apprentissage :",
    len(training_emissions)
)

print(
    "Matrice des émissions d'entraînement :",
    training_emissions.shape
)


# 5. Initialisation de la matrice de transition

transition_matrix = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90]
])


# L'état initial est alpha0

initial_probabilities = np.array([
    1.0,
    0.0,
    0.0
])


# 6. Baum-Welch

print("Début Baum-Welch sur les 10 000 régions...")

learned_transition_matrix = baum_welch(
    training_emissions,
    transition_matrix,
    initial_probabilities
)

print("Matrice de transition apprise :")
print(learned_transition_matrix)

# une fois le modèle appris on fait sur les données 
# 7. Forward-Backward final sur tous les bins candidats

print("Début du Forward-Backward final...")

forward, scaling = forward_algorithm(
    emissions,
    learned_transition_matrix,
    initial_probabilities
)

backward = backward_algorithm(
    emissions,
    learned_transition_matrix,
    scaling
)

# 8. Probabilités des états
probabilities = state_probabilities(
    forward,
    backward
)
# Diagnostic des probabilités des états

print("\nDIAGNOSTIC α2")

print(
    "Nombre avec P(α2) > 0.5 :",
    np.sum(probabilities[:, 2] > 0.5)
)

print(
    "Nombre avec P(α2) > 0.9 :",
    np.sum(probabilities[:, 2] > 0.9)
)

print(
    "Nombre avec P(α2) > 0.95 :",
    np.sum(probabilities[:, 2] > 0.95)
)


print("Matrice des probabilités d'états :", probabilities.shape)



# Chercher les bins où α2 est le plus probable
best_alpha2_indices = np.argsort(
    probabilities[:, 2]
)[-10:][::-1]

print("\n10 meilleurs bins pour α2 :")

for i in best_alpha2_indices:

    chromosome, start, x1, x2 = candidate_bins[i]

    print(
        chromosome,
        start,
        "ES =", x1,
        "NP =", x2,
        "Pα0 =", probabilities[i, 0],
        "Pα1 =", probabilities[i, 1],
        "Pα2 =", probabilities[i, 2]
    )


# 9. Identification des DHMS

rho = 0.95

dhms = []

for i, (chromosome, start, x1, x2) in enumerate(candidate_bins):

    p_alpha0 = probabilities[i, 0]
    p_alpha1 = probabilities[i, 1]
    p_alpha2 = probabilities[i, 2]

    if p_alpha1 > rho:
        state = "ESC"

    elif p_alpha2 > rho:
        state = "NPC"

    else:
        state = "non_differentiel"

    dhms.append(
        (
            chromosome,
            start,
            x1,
            x2,
            p_alpha0,
            p_alpha1,
            p_alpha2,
            state
        )
    )

print("Nombre total de bins candidats :", len(candidate_bins))

print(
    "Nombre de bins ESC-enrichis :",
    sum(1 for x in dhms if x[7] == "ESC")
)

print(
    "Nombre de bins NPC-enrichis :",
    sum(1 for x in dhms if x[7] == "NPC")
)

print(
    "Nombre de bins non différentiels :",
    sum(1 for x in dhms if x[7] == "non_differentiel")
)


 # fichier résultat
with open("resultats_HMM.tsv", "w") as out:

    out.write(
        "chrom\tstart\tES\tNP\t"
        "P_alpha0\tP_alpha1\tP_alpha2\tetat\n"
    )

    for row in dhms:

        out.write(
            f"{row[0]}\t"
            f"{row[1]}\t"
            f"{row[2]}\t"
            f"{row[3]}\t"
            f"{row[4]}\t"
            f"{row[5]}\t"
            f"{row[6]}\t"
            f"{row[7]}\n"
        )

print("Résultats sauvegardés dans resultats_HMM.tsv")


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