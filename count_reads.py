#print(bin_start)
from collections import defaultdict
import gzip

def count_reads(filename, bin_size=1000):

    # Ensemble permettant de supprimer les doublons
    # même position + même orientation
    unique_tags = set()

    # Dictionnaire contenant le nombre de fragments par bin
    counts = defaultdict(int)

    with gzip.open(filename, "rt") as f:

        for line in f:

            if line.startswith("#"):
                continue

            columns = line.strip().split("\t")

            chromosome = columns[0]
            position = int(columns[1])
            strand = columns[3]

            # Suppression des doublons
            key_tag = (chromosome, position, strand)

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

es_counts = count_reads(
    "Files/GSM307619_ES.H3K27me3.aligned.txt.gz"
)

np_counts = count_reads(
    "Files/GSM307614_NP.H3K27me3.aligned.txt.gz"
)