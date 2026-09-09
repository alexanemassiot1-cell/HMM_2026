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
) #table de probabilités d'émission

emissions = emissions_from_lookup(
    candidate_bins,
    emission_lookup
) #construis une matrice

print("Matrice des émissions :", emissions.shape)


# 4. Sélection aléatoire de 10 000 bins pour l'apprentissage

random.seed(42) # pour que le tirage soit reproductif 

n_training = 10000 #définis le nombre d'éléments utilisés pour l'apprentissage

training_indices = random.sample(
    range(len(candidate_bins)),
    n_training
) #tire aléatoirement 10 000 indices parmi tous les bins candidats.
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
]) # dans l'article alpha0 est fixé 


# 6. Baum-Welch

print("Début Baum-Welch sur les 10 000 régions...")

learned_transition_matrix = baum_welch(
    training_emissions,
    transition_matrix,
    initial_probabilities
)
#ajuster la matrice de transition jusqu'a qu'elle soient stable 

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
#quelle est la probabilité d'être dans chaque état en tenant compte de tout ce qui s'est passé avant le bin
backward = backward_algorithm(
    emissions,
    learned_transition_matrix,
    scaling
)
#quelle est la probabilité d'être dans chaque état en tenant compte des bins qui viennent après ?
# 8. Probabilités des états
probabilities = state_probabilities(
    forward,
    backward
)
# combine forward et backward 
# Chercher les bins où α2 est le plus probable
best_alpha2_indices = np.argsort( #Trie les indices selon leur probabilité.
    probabilities[:, 2] #probabilité que chaque bin soit NPC enrichi.
)[-10:][::-1] #Prend les 10 plus grandes valeurs et remet dans l'ordre décroissant.

print("\n10 meilleurs bins pour α2 :") #les 10 bins ayant les probabilités NPC les plus élevées.

for i in best_alpha2_indices:

    chromosome, start, x1, x2 = candidate_bins[i] # récupération des infos 

    print(
        chromosome,
        start,
        "ES =", x1,
        "NP =", x2,
        "Pα0 =", probabilities[i, 0],
        "Pα1 =", probabilities[i, 1],
        "Pα2 =", probabilities[i, 2]
    ) # verification du placement 


# 9. Identification des DHMS

rho = 0.95 # proba sup post à 95%

dhms = [] # list vide 

for i, (chromosome, start, x1, x2) in enumerate(candidate_bins): # parcours tout les candidats 

    p_alpha0 = probabilities[i, 0] # récupération de chaque état en fonction du candidats 
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
    ) # ajout de l'état 

print("Nombre total de bins candidats :", len(candidate_bins))

print(
    "Nombre de bins ESC-enrichis :",
    sum(1 for x in dhms if x[7] == "ESC") # comptage des résultats 
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
with open("resultats_HMM.tsv", "w") as out: # ouvre un fichier et écris 

    out.write(
        "chrom\tstart\tES\tNP\t"
        "P_alpha0\tP_alpha1\tP_alpha2\tetat\n"
    ) # première ligne du fichier 

    for row in dhms: # parcours tout les résultats 

        out.write(
            f"{row[0]}\t"
            f"{row[1]}\t"
            f"{row[2]}\t"
            f"{row[3]}\t"
            f"{row[4]}\t"
            f"{row[5]}\t"
            f"{row[6]}\t"
            f"{row[7]}\n"
        ) # put résultat dans le fichier 

print("Résultats sauvegardés dans resultats_HMM.tsv")


# 5. Fusion des bins DHMS consécutifs

def merge_dhms_bins(dhms): #sert à transformer plusieurs bins DHMS consécutifs en une seule région
    regions = [] # liste régions finales

    current_region = None #aucune région n'est en cours de construction

    for row in dhms: #parcourt tous les bins
        chromosome, start, x1, x2, p0, p1, p2, state = row #récupère les huit informations stockées

        # On ignore les bins non différentiels
        if state == "non_differentiel": # on enlève les bins non différenciels 
            if current_region is not None:
                regions.append(current_region) #on termine la région précédente si elle existe
                current_region = None
            continue

        end = start + 1000 # calcul la fin du bin car on sait qu'il font 1 kb 

        # Premier bin DHMS
        if current_region is None: # S'il n'existe aucune région en cours
            current_region = [ #on commence une nouvelle région
                chromosome,
                start,
                end,
                state
            ]
            continue

        current_chromosome, current_start, current_end, current_state = current_region

        # Le bin est-il directement adjacent au précédent ?
        if (
            chromosome == current_chromosome # meme chromosome
            and start == current_end #nvx bin ou commence la ou l'autre se termine
            and state == current_state # si meme états 
        ):
            # On prolonge la région si les cdt sont satifaite 
            current_region[2] = end

        else:
            # Nouvelle région
            regions.append(current_region) # sauvegarde la région précédente

            current_region = [ #démarre une nouvelle région
                chromosome,
                start,
                end,
                state
            ]

    # Ajouter la dernière région
    if current_region is not None:
        regions.append(current_region) # si une régions est en cours on ne la perd pas 

    return regions #renvoie toutes les régions fusionnées

# Fusion des bins DHMS
regions_dhms = merge_dhms_bins(dhms) # on fait cette fonction à tout les bins

print("\nAprès fusion des bins DHMS :")
print("Nombre total de régions :", len(regions_dhms)) #Compte les régions

n_esc_regions = sum(
    1 for region in regions_dhms
    if region[3] == "ESC"
) # compte les ESC

n_npc_regions = sum(
    1 for region in regions_dhms
    if region[3] == "NPC"
) # compte les NPC

print("Nombre de régions ESC :", n_esc_regions)
print("Nombre de régions NPC :", n_npc_regions)

with open("regions_DHMS.tsv", "w") as out: # création du fichier finales 
    out.write("chrom\tstart\tend\tetat\n") # en tete du fichier 

    for chromosome, start, end, state in regions_dhms: # parcours toute les régions 
        out.write( # ecrit pout chaque régions 
            f"{chromosome}\t"
            f"{start}\t"
            f"{end}\t"
            f"{state}\n"
        )

print("Régions sauvegardées dans regions_DHMS.tsv")
