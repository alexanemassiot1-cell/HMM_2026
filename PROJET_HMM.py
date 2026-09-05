# PROJET_HMM
""" importation des données """
import gzip
from collections import defaultdict


filename = "Files/GSM307619_ES.H3K27me3.aligned.txt.gz"

with gzip.open(filename, "rt") as f:

    for line in f:
        columns = line.strip().split("\t")

        chromosome = columns[0]
        start = int(columns[1])
        end = int(columns[2])

        print(chromosome, start, end)

        break

#Calcul de bins de 1 kB
bin_start = (start // 1000) * 1000

#print(bin_start)

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


all_bins = set(es_counts) | set(np_counts) #prendre tous les bins qui existent dans ES OU dans NP.

# Nombre total de fragments
n1 = sum(es_counts.values())
n2 = sum(np_counts.values())

# création d'un fichier qui va pouvoir etre analyser en 1kb 

with open("input_HMM.tsv", "w") as out: 

    out.write("chrom\tstart\tend\tES\tNP\tF\n") # \t correspond a des tabulations 

    for chromosome, start in sorted(all_bins): # on parcourt tout les bins sorted permet de les trier avant de les parcourirs

        end = start + 1000 # chaque bin correspond à 1000 paire de base 

        es = es_counts.get((chromosome, start), 0) # dictionnaire  pour récupéré le es de charque chrs
        np = np_counts.get((chromosome, start), 0) # dictionnaire pour récupéré le np de charque chrs

        # Calcul du score F(i)
        F = (es / n1) + (np / n2) # pour chaque bins 

        out.write(
            f"{chromosome}\t"
            f"{start}\t"
            f"{end}\t"
            f"{es}\t"
            f"{np}\n"
            f"{F}\n"
        )


# Nombre total de bins
m = len(all_bins)

# Paramètre de l'article
eta = 0.7

# Seuil
threshold = 2 / (m * eta)

# Paramètres du prior Beta
alpha = 1
beta = m

def posterior_mean(x, n):
    return (alpha + x) / (alpha + beta + n)


#sans recreer une data en partant du fichier créer resultat
candidate_bins = []


for chromosome, start in sorted(all_bins):

    end = start + 1000

    es = es_counts.get((chromosome, start), 0)
    np = np_counts.get((chromosome, start), 0)

    F = (es / n1) + (np / n2)

    if F > threshold:
        candidate_bins.append(
            (chromosome, start, end, es, np, F)
        )


"""
Si une fenêtre génomique a une certaine probabilité \(p\) 
de recevoir un fragment, quelle est la probabilité d'observer exactement \(x\) 
fragments dans cette fenêtre ?
Dans ton projet ChIP-seq, c'est utile parce que les fragments sont issus d'un échantillonnage. 
Le nombre de fragments que tu observes dans une fenêtre peut donc varier simplement à cause du hasard.

le modèle binomiale + beta aide a HMM de prendre la décission dans quelle état c'est alpha 0, 1 ou 2
"""

for chromosome, start in sorted(all_bins):

    x_es = es_counts.get((chromosome, start), 0)
    x_np = np_counts.get((chromosome, start), 0)

    p_es = posterior_mean(x_es, n1)
    p_np = posterior_mean(x_np, n2)

    print(chromosome, start, x_es, x_np, p_es, p_np)


# Commencement de HMM
"""
probabilité d'emission sert a savoir si c'est dans alpha0 ou alpha 1 ou 2 
"""
from math import comb # comb sert a calculer le coefficient binomiale 
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

def log_binomial_probability(x, n, p):
    """
    Logarithme de la probabilité binomiale :
    
    P(X=x | p) = C(n,x) * p^x * (1-p)^(n-x)
    """
    
    if p <= 0:
        if x == 0:
            return 0.0
        else:
            return -np.inf

    if p >= 1:
        if x == n:
            return 0.0
        else:
            return -np.inf

    log_comb = (
        gammaln(n + 1)
        - gammaln(x + 1)
        - gammaln(n - x + 1)
    )

    return (
        log_comb
        + x * np.log(p)
        + (n - x) * np.log1p(-p)
    )
""" 
x = nombre de fragments observés dans une fenêtre
n = nombre total de fragments de la bibliothèque
p = probabilité qu'un fragment tombe dans cette fenêtre

Pourquoi \(1-p\) ?

Parce qu'on distingue :

\(p\) = probabilité qu'un fragment tombe dans la fenêtre
\(1-p\) = probabilité qu'un fragment ne tombe pas dans la fenêtre

Si on observe \(x\) fragments dans la fenêtre sur \(n\) fragments au total, alors :

\(x\) sont dans la fenêtre → \(p^x\)
\(n-x\) ne sont pas dans la fenêtre → (1−p)
n−x
Quelle est la probabilité d'obtenir \(x\) fragments dans cette fenêtre sachant \(p\) ?
"""
import numpy as np
from scipy.special import beta
from scipy.special import gammaln, betaln, betainc
from scipy.integrate import quad

def log_beta_prior(p, alpha, beta_param):
    """
    Logarithme de la densité Beta.
    """

    if p <= 0 or p >= 1:
        return -np.inf

    # alpha = 1 dans notre cas
    return (
        (alpha - 1) * np.log(p)
        + (beta_param - 1) * np.log1p(-p)
    )
"""
p valeur possible de l'intensité 
La fonction nous dit alors à quel point cette valeur de \(p\) est plausible selon notre distribution a priori.
calcule la fonction Beta \(B(\alpha,\beta)\).

Elle sert à normaliser la distribution pour que l'ensemble des probabilités sur \(0\leq p\leq1\) soit cohérent.

la fonction beta importer qui fait la normalisation grace à ca : scipy.special.beta . 
"""
def state_constraint(p1, p2, state, tau):

    if p2 <= 0:
        return False

    ratio = p1 / p2

    if state == 0:
        # α0 : non différentiel
        return (1 / tau) <= ratio <= tau

    elif state == 1:
        # α1 : enrichi dans ESC
        return ratio > tau

    elif state == 2:
        # α2 : enrichi dans NPC
        return ratio < (1 / tau)

    return False


def emission_probability(x1, x2, n1, n2, m, tau, state):

    alpha = 1
    beta_param = m

    # Paramètres de la loi Beta posterior
    A1 = alpha + x1
    B1 = beta_param + n1 - x1

    A2 = alpha + x2
    B2 = beta_param + n2 - x2

    # Coefficients binomiaux sous forme logarithmique
    log_C1 = (
        gammaln(n1 + 1)
        - gammaln(x1 + 1)
        - gammaln(n1 - x1 + 1)
    )

    log_C2 = (
        gammaln(n2 + 1)
        - gammaln(x2 + 1)
        - gammaln(n2 - x2 + 1)
    )

    # --------------------------------------------------
    # Bornes possibles de p1 en fonction de p2
    # --------------------------------------------------

    def p1_bounds(p2):

        if state == 0:
            # 1/tau <= p1/p2 <= tau
            low = p2 / tau
            high = min(tau * p2, 1.0)

        elif state == 1:
            # p1/p2 > tau
            low = tau * p2
            high = 1.0

        elif state == 2:
            # p1/p2 < 1/tau
            low = 0.0
            high = p2 / tau

        else:
            raise ValueError("state doit être 0, 1 ou 2")

        if low >= high:
            return None

        return low, high

    # --------------------------------------------------
    # Intégrale sur p2 pour le numérateur
    # --------------------------------------------------

    def numerator_integrand(p2):

        bounds = p1_bounds(p2)

        if bounds is None:
            return 0.0

        low, high = bounds

        if high <= 0:
            return 0.0

        # Intégrale de la partie p1 :
        #
        # ∫ p1^(A1-1) (1-p1)^(B1-1) dp1
        #
        # calculée avec la fonction Beta incomplète.

        integral_p1 = (
            np.exp(betaln(A1, B1))
            * (
                betainc(A1, B1, high)
                - betainc(A1, B1, low)
            )
        )

        if integral_p1 <= 0:
            return 0.0

        # Partie p2
        p2_part = (
            p2 ** (A2 - 1)
            * (1 - p2) ** (B2 - 1)
        )

        return p2_part * integral_p1

    # --------------------------------------------------
    # Numérateur
    # --------------------------------------------------

    numerator, _ = quad(
        numerator_integrand,
        1e-12,
        1.0,
        epsabs=1e-15,
        epsrel=1e-6,
        limit=200
    )

    # --------------------------------------------------
    # Dénominateur
    # --------------------------------------------------

    def denominator_integrand(p2):

        bounds = p1_bounds(p2)

        if bounds is None:
            return 0.0

        low, high = bounds

        integral_p1 = (
            np.exp(betaln(alpha, beta_param))
            * (
                betainc(
                    alpha,
                    beta_param,
                    high
                )
                -
                betainc(
                    alpha,
                    beta_param,
                    low
                )
            )
        )

        p2_part = (
            p2 ** (alpha - 1)
            * (1 - p2) ** (beta_param - 1)
        )

        return p2_part * integral_p1

    denominator, _ = quad(
        denominator_integrand,
        1e-12,
        1.0,
        epsabs=1e-15,
        epsrel=1e-6,
        limit=200
    )

    if denominator <= 0 or numerator <= 0:
        return 0.0

    # Les constantes binomiales
    # doivent être réintroduites ici.
    log_result = (
        log_C1
        + log_C2
        + np.log(numerator)
        - np.log(denominator)
    )

    # Évite overflow/underflow
    if log_result < -745:
        return 0.0

    if log_result > 700:
        return np.inf

    return np.exp(log_result)

"""
Contraite en fonction de ce que l'on veut voir

"""
#fait pour une fenettre car on a beaucoup de fenettre donc juste pour tester si ca marche sur une êtite fenetre 

x_es = 50
x_np = 10

n1 = sum(es_counts.values())
n2 = sum(np_counts.values())

m = len(all_bins)

tau = 3.0

e0 = emission_probability(
    x_es, x_np, n1, n2, m, tau, 0
)

e1 = emission_probability(
    x_es, x_np, n1, n2, m, tau, 1
)

e2 = emission_probability(
    x_es, x_np, n1, n2, m, tau, 2
)

print("Emission α0 :", e0)
print("Emission α1 :", e1)
print("Emission α2 :", e2)

print("Emission α0 :", e0)
print("Emission α1 :", e1)
print("Emission α2 :", e2)
