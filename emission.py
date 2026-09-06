import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.special import betaln, gammaln

# Commencement de HMM
"""
probabilité d'emission sert a savoir si c'est dans alpha0 ou alpha 1 ou 2 
"""
from math import comb # comb sert a calculer le coefficient binomiale 
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from math import comb

def binomial_probability(x, n, p):
    return (
        comb(n, x)
        * p**x
        * (1-p)**(n-x)
    )


"""
x1 : nombre de fragments ESC dans la fenêtre
x2 : nombre de fragments NPC dans la fenêtre
n1 : nombre total de fragments ESC
n2 : nombre total de fragments NPC
m : nombre de fenêtres du génome
tau : seuil utilisé pour définir les états
etat : état considéré : 0, 1 ou 2

"""
def emission_probability(
    x1, x2,
    n1, n2,
    a1, b1,
    a2, b2,
    m, tau, etat
):
# parametre calculer dans la fonction postérior.mean
    alpha = 1
    beta_param = m

    # 1. Probabilité des observations 
    """ 
    gammaln : permet de calculer les factorielles sous forme logarithmique.
    betaln : permet de calculer la distribution des bibliothèques ESC/NP sans manipuler des nombres très petit
    """

    # ESC
    log_marginal_1 = (
        gammaln(n1 + 1)
        - gammaln(x1 + 1)
        - gammaln(n1 - x1 + 1)
        + betaln(a1, b1)
        - betaln(alpha, beta_param)
    )

    # NPC
    log_marginal_2 = (
        gammaln(n2 + 1)
        - gammaln(x2 + 1)
        - gammaln(n2 - x2 + 1)
        + betaln(a2, b2)
        - betaln(alpha, beta_param)
    )

    # 2. Fonction donnant les limites de p1

    def p1_bounds(p2): # sert à déterminer les valeurs de p_1 qui sont autorisées pour un état donné du HMM.

        if etat == 0:
            # α0 : non différentiel
            # 1/tau <= p1/p2 <= tau = 3 définit à partir de quel rapport on considère que les deux intensités sont différentes.

            low = p2 / tau
            high = min(tau * p2, 1.0)

        elif etat == 1:
            # α1 : enrichi dans ESC
            # p1/p2 > tau

            low = tau * p2
            high = 1.0

        elif etat == 2:
            # α2 :enrichi dans NPC
            # p1/p2 < 1/tau

            low = 0.0
            high = p2 / tau

        else:
            raise ValueError(
                "Etat doit être 0, 1 ou 2"
            )

        if low >= high:
            return None

        return low, high

    # 3. Probabilité que (p1,p2) appartienne
    #    à la région de l'état de HMM considéré 


    def region_probability(a1, b1, a2, b2): 

        def integrand(q): # On va intégrer sur toutes les valeurs possibles de p2
	

            # q est une probabilité cumulée pour p2
            # On récupère donc la valeur correspondante
            p2 = beta_dist.ppf(q, a2, b2) #

            if not np.isfinite(p2):
                return 0.0

            bounds = p1_bounds(p2) # on cherche les valeurs autorisé

            if bounds is None:
                return 0.0

            low, high = bounds

            # Probabilité que p1 soit dans [low, high]
            probability_p1 = (
                beta_dist.cdf(high, a1, b1)
                - beta_dist.cdf(low, a1, b1)
            )

            return probability_p1

        # On évite exactement 0 et 1
        """
        pour ca il fait une intégrale :
        Intègre numériquement integrand pour toutes les valeurs possibles de q entre 0 et 1, 
        avec une certaine précision, puis retourne le résultat de cette intégrale.
        """
        eps = 1e-10 #Ne commence pas exactement à 0 et ne termine pas exactement à 1.
        # Quelle est la proportion des valeurs possibles de p1 et p2 qui correspondent à l'état que je suis en train d'étudier ?
        """quad est juste un outil qui fait une addition très précise.
"""
        result, error = quad(
            integrand,
            eps,
            1 - eps,
            epsabs=1e-8,
            epsrel=1e-6,
            limit=100
        )
        #Prends ma fonction integrand et calcule sa somme sur toutes les valeurs entre presque 0 et presque 1.

        return result

    # 4. Région sous le posterior

    posterior_region = region_probability(
        a1, b1,
        a2, b2
    )
    # avant d'observer les echantillions 

    # 5. Région sous le prior
    # vérification de sécurité

    prior_region = region_probability(
        alpha, beta_param,
        alpha, beta_param
    )

    if posterior_region <= 0:
        return 0.0

    if prior_region <= 0:
        return 0.0
    

    # 6. Probabilité d'émission finale
    """ log_marginal_1 : Est-ce que les données observées en ESC sont compatibles avec les intensités que le modèle considère ?
        log_marginal_2 : Est-ce que les données observées en NPS sont compatibles avec les intensités que le modèle considère ?
    """

    log_emission = (
        log_marginal_1
        + log_marginal_2
        + np.log(posterior_region) #a probabilité d'être dans la région correspondant à l'état après avoir observé les données.
        - np.log(prior_region) #prior_region représente la même chose mais avant d'observer les données.
    )
    # on compare les données avant de les avoir avec ceux d'après 
    # le log est pour éviter d'avoir des proba trop petites 

    # Évite l'underflow
    if log_emission < -745:
        return 0.0
    # PROTECTION informatique Si le log est inférieur à -745, la probabilité est tellement petite qu'on la considère comme 0.
    return np.exp(log_emission) # exp pour enlever le log après les calcules

#test sur une fenettre
x_es = 30
x_np = 2

n1 = 1000
n2 = 1000

m = 1000
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