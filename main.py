from count_reads import count_reads
from emission import build_emission_lookup, emissions_from_lookup
from baum_welch import baum_welch
from etat_transission import (
    forward_algorithm,
    backward_algorithm,
    state_probabilities
)

import numpy as np
import random

# 1. Lecture et comptage
es_counts = count_reads("Files/GSM307619_ES.H3K27me3.aligned.txt.gz") # appel des fichier du comptage pour ESC
np_counts = count_reads("Files/GSM307614_NP.H3K27me3.aligned.txt.gz")# appel des fichier du comptage pour NPC
# 2. Création des bins
all_bins = set(es_counts) | set(np_counts) # on unie les deux ensemble car un bin peut etre ESC et NPC
# ensuite on veut conservé le bin  si il est au moins dans un des deux 
bins = [] 

for chromosome, start in sorted(all_bins): # on parcours tout les bins dans l'ordre genomique d'ou le sort

    x1 = es_counts.get((chromosome, start), 0) # pour chaque bin on met le nombre de ESC
    x2 = np_counts.get((chromosome, start), 0)# pour chaque bin on met le nombre de NPC
    # si aucun bin est présent ca renvoie 0

    bins.append(
        (chromosome, start, x1, x2)
    )# on le stock dans une autre variable initier précedemment

# 3. Paramètres globaux
n1 = sum(es_counts.values())# nombre total de ESC après le prétraitement
n2 = sum(np_counts.values())# nombre total de NPC après le prétraitement
# cela servira à normaliser par la suite 
m = len(all_bins) #représente donc le nombre de bins présents dans tes données.

tau = 3.0 #la différence minimale entre les deux intensités pour considérer plus enrichie que l'autre.
eta = 0.7 # sélections des candidats 


print("Nombre total de fragments ESC :", n1)
print("Nombre total de fragments NPC :", n2)
print("Nombre de bins :", m)

# Calcul de F(i) et sélection des bins candidats
threshold = 2 / (m * eta) #C'est le seuil utilisé pour F(i). 
# Car on veut éliminer les bins avec très peu de signal.
candidate_bins = []

for chromosome, start, x1, x2 in bins:

    F = (x1 / n1) + (x2 / n2)

    if F > threshold:

        candidate_bins.append(
            (chromosome, start, x1, x2)
        )

# calcul de F(i) 
"""C'est une étape de filtrage avant le HMM.
Le HMM ne peut pas travailler directement sur tous les bins du génome."""




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

random.seed(1)

n_training = 10000

training_indices = random.sample(
    range(len(candidate_bins)),
    n_training
)
training_indices.sort() # Les régions sont sélectionnées aléatoirement, puis remises dans l'ordre génomique avant l'apprentissage.
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


# 5. Fusion des bins DHMS consécutifs

def merge_dhms_bins(dhms):
    regions = []

    current_region = None

    for row in dhms:
        chromosome, start, x1, x2, p0, p1, p2, state = row

        # On ignore les bins non différentiels
        if state == "non_differentiel":
            if current_region is not None:
                regions.append(current_region)
                current_region = None
            continue

        end = start + 1000

        # Premier bin DHMS
        if current_region is None:
            current_region = [
                chromosome,
                start,
                end,
                state
            ]
            continue

        current_chromosome, current_start, current_end, current_state = current_region

        # Le bin est-il directement adjacent au précédent ?
        if (
            chromosome == current_chromosome
            and start == current_end
            and state == current_state
        ):
            # On prolonge la région
            current_region[2] = end

        else:
            # Nouvelle région
            regions.append(current_region)

            current_region = [
                chromosome,
                start,
                end,
                state
            ]

    # Ajouter la dernière région
    if current_region is not None:
        regions.append(current_region)

    return regions

# Fusion des bins DHMS
regions_dhms = merge_dhms_bins(dhms)

print("\nAprès fusion des bins DHMS :")
print("Nombre total de régions :", len(regions_dhms))

n_esc_regions = sum(
    1 for region in regions_dhms
    if region[3] == "ESC"
)

n_npc_regions = sum(
    1 for region in regions_dhms
    if region[3] == "NPC"
)

print("Nombre de régions ESC :", n_esc_regions)
print("Nombre de régions NPC :", n_npc_regions)

with open("regions_DHMS.tsv", "w") as out:
    out.write("chrom\tstart\tend\tetat\n")

    for chromosome, start, end, state in regions_dhms:
        out.write(
            f"{chromosome}\t"
            f"{start}\t"
            f"{end}\t"
            f"{state}\n"
        )

print("Régions sauvegardées dans regions_DHMS.tsv")
