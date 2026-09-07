#print(bin_start)
from collections import defaultdict
import gzip # décompresse les zip

def count_reads(filename, bin_size=1000): # fichier + tailles des fenetres génomique

    # Ensemble permettant de supprimer les doublons
    # même position + même orientation et mémorise les tags
    unique_tags = set()

    # Dictionnaire contenant le nombre de fragments par bin
    counts = defaultdict(int) # qui va donc compter 

    with gzip.open(filename, "rt") as f: # permet de le lire directement

        for line in f: # ligne par ligne

            if line.startswith("#"): # on supprime les lignes qui commencent par #
                continue

            columns = line.strip().split("\t") # on sépare pour les mettres en colonnes 

            chromosome = columns[0]
            position = int(columns[1])
            strand = columns[3]

            # Suppression des doublons
            key_tag = (chromosome, position, strand) # 2 reads sont identiques si tout ses trucs sont identiques

            if key_tag in unique_tags:
                continue

            unique_tags.add(key_tag)

            # Estimation du centre du fragment
            if strand == "+":
                fragment_center = position + 100
            else:
                fragment_center = position - 100

            # Attribution à un bin de 1 kb
            bin_start = (fragment_center // bin_size) * bin_size

            key_bin = (chromosome, bin_start)

            counts[key_bin] += 1

    return counts

