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

es_counts = count_reads(
    "Files/GSM307619_ES.H3K27me3.aligned.txt.gz"
)  # appel du fichier du comptage pour ESC

np_counts = count_reads(
    "Files/GSM307614_NP.H3K27me3.aligned.txt.gz"
)  # appel du fichier du comptage pour NPC


# 2. Création des bins

all_bins = set(es_counts) | set(np_counts)
# on unit les deux ensembles car un bin peut être présent en ESC et NPC

# ensuite on veut conserver le bin s'il est au moins dans un des deux

bins = []

for chromosome, start in sorted(all_bins):
    # on parcourt tous les bins dans l'ordre génomique d'où le sorted

    x1 = es_counts.get(
        (chromosome, start),
        0
    )  # pour chaque bin on met le nombre de fragments ESC

    x2 = np_counts.get(
        (chromosome, start),
        0
    )  # pour chaque bin on met le nombre de fragments NPC

    # si aucun bin n'est présent dans une condition, cela renvoie 0

    bins.append(
        (chromosome, start, x1, x2)
    )  # on le stocke dans une autre variable initialisée précédemment


# 3. Paramètres globaux

n1 = sum(es_counts.values())
# nombre total de fragments ESC après le prétraitement

n2 = sum(np_counts.values())
# nombre total de fragments NPC après le prétraitement

# cela servira à normaliser par la suite

m = len(all_bins)
# représente donc le nombre de bins présents dans tes données

tau = 3.0
# la différence minimale entre les deux intensités pour considérer
# une condition comme plus enrichie que l'autre

eta = 0.7
# sélection des candidats


print("Nombre total de fragments ESC :", n1)

print("Nombre total de fragments NPC :", n2)

print("Nombre de bins :", m)


# Calcul de F(i) et sélection des bins candidats

threshold = 2 / (m * eta)
# c'est le seuil utilisé pour F(i)

# car on veut éliminer les bins avec très peu de signal

candidate_bins = []

for chromosome, start, x1, x2 in bins:

    F = (x1 / n1) + (x2 / n2)
    # calcul du score F(i)

    if F > threshold:

        candidate_bins.append(
            (chromosome, start, x1, x2)
        )


# C'est une étape de filtrage avant le HMM.
# Le HMM ne peut pas travailler directement sur tous les bins du génome.


print(
    "Nombre de bins candidats :",
    len(candidate_bins)
)


# 4. Regroupement des bins candidats en régions candidates

def merge_candidate_bins(candidate_bins):
    # sert à transformer plusieurs bins candidats consécutifs
    # en une seule région candidate

    regions = []
    # liste des régions candidates

    current_region = None
    # aucune région n'est en cours de construction

    for row in candidate_bins:
        # parcourt tous les bins candidats

        chromosome, start, x1, x2 = row
        # récupère les informations du bin candidat

        end = start + 1000
        # calcul de la fin du bin car on sait qu'il fait 1 kb

        if current_region is None:
            # s'il n'existe aucune région en cours

            current_region = [
                chromosome,
                start,
                end,
                [row]
            ]
            # on commence une nouvelle région

            continue

        current_chromosome, current_start, current_end, current_bins = current_region
        # récupère les informations de la région en cours

        # le bin est-il directement adjacent au précédent ?

        if (
            chromosome == current_chromosome
            # même chromosome

            and start == current_end
            # le nouveau bin commence là où l'autre se termine
        ):

            current_region[2] = end
            # on prolonge la région

            current_bins.append(row)
            # on ajoute le bin à la région

        else:

            regions.append(current_region)
            # sauvegarde la région précédente

            current_region = [
                chromosome,
                start,
                end,
                [row]
            ]
            # démarre une nouvelle région

    # ajouter la dernière région

    if current_region is not None:

        regions.append(current_region)
        # si une région est en cours, on ne la perd pas

    return regions
    # renvoie toutes les régions candidates


candidate_regions = merge_candidate_bins(candidate_bins)
# on regroupe les bins candidats afin d'obtenir les régions candidates


print(
    "Nombre de régions candidates :",
    len(candidate_regions)
)


# 5. Construction de la lookup table

print("Construction de la lookup table...")

emission_lookup = build_emission_lookup(
    candidate_bins,
    n1,
    n2,
    m,
    tau
)  # table de probabilités d'émission


emissions = emissions_from_lookup(
    candidate_bins,
    emission_lookup
)  # construit une matrice


print("Matrice des émissions :", emissions.shape)

# Fusion des bins candidats pour créer les régions candidates



# 4. Sélection aléatoire de 10 000 régions pour l'apprentissage

random.seed(42)  # permet d'avoir toujours le même tirage aléatoire à chaque exécution
# sans cette ligne, les 10 000 régions choisies pourraient changer à chaque lancement


n_training = 10000  # définit le nombre de régions que l'on veut utiliser pour l'apprentissage
# l'article utilise un sous-ensemble aléatoire de 10 000 régions


training_region_indices = random.sample(
    range(len(candidate_regions)),
    n_training
)
# tire aléatoirement 10 000 indices parmi toutes les régions candidates
# par exemple : [4, 15, 28, 42, ...]
# chaque indice correspond à une région dans candidate_regions


# On récupère les régions sélectionnées aléatoirement

training_regions = [
    candidate_regions[i]
    for i in training_region_indices
]
# utilise les indices précédemment sélectionnés pour récupérer
# les régions correspondantes dans candidate_regions
# training_regions contient maintenant les 10 000 régions choisies


print(
    "Nombre de régions utilisées pour l'apprentissage :",
    len(training_regions)
)
# vérifie que l'on a bien sélectionné 10 000 régions

# Fusion des bins DHMS
def merge_dhms_bins(dhms):
    # sert à transformer plusieurs bins DHMS consécutifs
    # en une seule région

    regions = []

    # liste qui contiendra les régions finales

    current_region = None

    # aucune région DHMS n'est encore en cours de construction

    for row in dhms:

        # on parcourt tous les bins classés par le HMM

        chromosome, start, x1, x2, p0, p1, p2, state = row

        # on récupère les informations du bin :
        # chromosome, position, nombres de fragments,
        # probabilités des 3 états et état final

        if state == "non_differentiel":

            # si le bin n'est pas différentiel,
            # il ne doit pas faire partie d'une région DHMS

            if current_region is not None:

                # si une région était en cours,
                # on la sauvegarde

                regions.append(current_region)

                current_region = None

                # puis on indique qu'aucune région n'est en cours

            continue

            # on passe au bin suivant

        end = start + 1000

        # chaque bin fait 1 kb = 1000 pb
        # donc sa position de fin est start + 1000

        if current_region is None:

            # si aucune région n'est en cours,
            # on commence une nouvelle région

            current_region = [
                chromosome,
                start,
                end,
                state
            ]

            continue

        current_chromosome, current_start, current_end, current_state = current_region

        # on récupère les informations de la région actuelle

        if (
            chromosome == current_chromosome
            and start == current_end
            and state == current_state
        ):

            # même chromosome
            # ET bin directement adjacent
            # ET même état (ESC ou NPC)

            current_region[2] = end

            # on agrandit la région jusqu'à la fin du nouveau bin

        else:

            # sinon, le nouveau bin ne peut pas être fusionné
            # avec la région actuelle

            regions.append(current_region)

            # on sauvegarde donc la région précédente

            current_region = [
                chromosome,
                start,
                end,
                state
            ]

            # puis on commence une nouvelle région

    # après avoir parcouru tous les bins,
    # il peut rester une dernière région en cours

    if current_region is not None:
        regions.append(current_region)

    # on renvoie toutes les régions DHMS fusionnées

    return regions

# Création d'un dictionnaire permettant de retrouver
# rapidement l'indice de chaque bin candidat

candidate_bin_indices = {
    (chromosome, start): i
    for i, (chromosome, start, x1, x2)
    in enumerate(candidate_bins)
}
# chaque bin candidat possède un chromosome et une position de départ
# on associe donc chaque couple (chromosome, start) à son indice
# cela permet ensuite de retrouver rapidement un bin dans candidate_bins


# On construit une matrice des émissions séparée
# pour chaque région

training_emissions_regions = []
# liste vide qui va contenir les émissions de chaque région
# chaque élément de cette liste correspondra à UNE région
#
# contrairement à training_emissions,
# on ne va pas mettre toutes les régions bout à bout
# chaque région restera une séquence indépendante


for region in training_regions:
    # on parcourt les 10 000 régions sélectionnées une par une


    chromosome = region[0]
    # récupère le chromosome sur lequel se trouve la région
    # exemple : chr1


    start = region[1]
    # récupère la position de début de la région
    # exemple : 35000000


    end = region[2]
    # récupère la position de fin de la région
    # exemple : 35003000


    # Les bins appartenant à cette région

    region_indices = []
    # liste vide qui va contenir les indices des bins
    # appartenant à la région actuelle


    current_start = start
    # commence au premier bin de la région


    while current_start < end:
        # on parcourt la région bin par bin
        # les bins ont une taille de 1 kb = 1000 pb


        key = (
            chromosome,
            current_start
        )
        # crée la clé permettant d'identifier le bin
        # exemple : ("chr1", 35000000)


        if key in candidate_bin_indices:
            # vérifie si ce bin existe bien dans les bins candidats


            region_indices.append(
                candidate_bin_indices[key]
            )
            # récupère l'indice du bin dans candidate_bins
            # puis ajoute cet indice à region_indices


        current_start += 1000
        # passe au bin suivant
        # comme les bins font 1000 pb,
        # on avance de 1000 bases à chaque étape


    # On récupère les émissions correspondant
    # aux bins de cette région

    if len(region_indices) > 0:
        # vérifie que la région contient bien au moins un bin candidat


        region_emissions = emissions[
            region_indices
        ]
        # récupère dans la matrice emissions
        # les probabilités d'émission correspondant aux bins de cette région
        #
        # chaque ligne correspond à un bin
        # et les 3 colonnes correspondent aux 3 états :
        # α0 = non différentiel
        # α1 = ESC
        # α2 = NPC


        training_emissions_regions.append(
            region_emissions
        )
        # ajoute la matrice d'émissions de cette région
        # à la liste des régions d'apprentissage
        #
        # on garde donc chaque région séparée


# Nombre total de bins appartenant aux régions d'apprentissage

n_training_bins = sum(
    len(region)
    for region in training_emissions_regions
)
# compte le nombre total de bins contenus dans les 10 000 régions
#
# par exemple :
# 10 000 régions peuvent contenir 18 946 bins
# car certaines régions possèdent plusieurs bins


print(
    "Nombre de bins appartenant aux régions d'apprentissage :",
    n_training_bins
)
# affiche le nombre total de bins utilisés pour l'apprentissage


print(
    "Nombre de séquences d'entraînement :",
    len(training_emissions_regions)
)
# affiche le nombre de régions réellement utilisées comme séquences
# on doit normalement obtenir 10 000



# 7. Initialisation de la matrice de transition

transition_matrix = np.array([
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3]
])

# initialisation de la matrice uniformément comme dans l'article


# L'état initial est alpha0

initial_probabilities = np.array([
    1.0,
    0.0,
    0.0
])

# dans l'article alpha0 est fixé


# 8. Baum-Welch

print(
    "Début Baum-Welch sur les 10 000 régions..."
)

learned_transition_matrix = baum_welch(
    training_emissions_regions,
    transition_matrix,
    initial_probabilities
)

# ajuster la matrice de transition jusqu'à ce qu'elle soit stable

print("Matrice de transition apprise :")
print(learned_transition_matrix)


# une fois le modèle appris on fait l'analyse sur les données


# 9. Forward-Backward final sur tous les bins candidats

print("Début du Forward-Backward final...")

forward, scaling = forward_algorithm(
    emissions,
    learned_transition_matrix,
    initial_probabilities
)

# quelle est la probabilité d'être dans chaque état
# en tenant compte de tout ce qui s'est passé avant le bin

backward = backward_algorithm(
    emissions,
    learned_transition_matrix,
    scaling
)

# quelle est la probabilité d'être dans chaque état
# en tenant compte des bins qui viennent après ?


# 10. Probabilités des états

probabilities = state_probabilities(
    forward,
    backward
)

# combine forward et backward


# Chercher les bins où α2 est le plus probable

best_alpha2_indices = np.argsort(
    probabilities[:, 2]
)[-10:][::-1]

# Trie les indices selon leur probabilité.
# Prend les 10 plus grandes valeurs et remet dans l'ordre décroissant.


print("\n10 meilleurs bins pour α2:")

# les 10 bins ayant les probabilités NPC les plus élevées.

for i in best_alpha2_indices:

    chromosome, start, x1, x2 = candidate_bins[i]

    # récupération des infos

    print(
        chromosome,
        start,
        "ES =", x1,
        "NP =", x2,
        "Pα0 =", probabilities[i, 0],
        "Pα1 =", probabilities[i, 1],
        "Pα2 =", probabilities[i, 2]
    )

    # verification du placement


# 11. Identification des DHMS

rho = 0.95  # proba sup post à 95%

dhms = []  # list vide

for i, (chromosome, start, x1, x2) in enumerate(candidate_bins):

    # parcours tout les candidats

    p_alpha0 = probabilities[i, 0]

    # récupération de chaque état en fonction du candidats

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

    # ajout de l'état


print(
    "Nombre total de bins candidats :",
    len(candidate_bins)
)

print(
    "Nombre de bins ESC-enrichis :",
    sum(row[-1] == "ESC" for row in dhms)
)

print(
    "Nombre de bins NPC-enrichis :",
    sum(row[-1] == "NPC" for row in dhms)
)

print(
    "Nombre de bins non différentiels :",
    sum(row[-1] == "non_differentiel" for row in dhms)
)


# Sauvegarde des résultats

with open("resultats_HMM.tsv", "w") as f:

    f.write(
        "chromosome\tstart\tES\tNP\tPalpha0\tPalpha1\tPalpha2\tstate\n"
    )

    for row in dhms:

        f.write(
            "\t".join(map(str, row))
            + "\n"
        )

print(
    "Résultats sauvegardés dans resultats_HMM.tsv"
)


# Fusion des bins DHMS

regions_dhms = merge_dhms_bins(dhms)

print("\nAprès fusion des bins DHMS:")

print(
    "Nombre total de régions :",
    len(regions_dhms)
)  # Compte les régions

print(
    "Nombre de régions ESC :",
    sum(region[3] == "ESC" for region in regions_dhms)
)

print(
    "Nombre de régions NPC :",
    sum(region[3] == "NPC" for region in regions_dhms)
)


# Sauvegarde des régions

with open("regions_DHMS.tsv", "w") as f:

    f.write(
        "chromosome\tstart\tend\tstate\n"
    )

    for region in regions_dhms:

        f.write(
            "\t".join(map(str, region))
            + "\n"
        )

print(
    "Régions sauvegardées dans regions_DHMS.tsv"
)