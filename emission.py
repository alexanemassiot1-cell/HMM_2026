import numpy as np

from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.special import betaln, gammaln

from posterior import posterior_parameters
import warnings


# ============================================================
# 1. Probabilité binomiale
# ============================================================

def binomial_probability(x, n, p):

    if p <= 0:
        return 1.0 if x == 0 else 0.0

    if p >= 1:
        return 1.0 if x == n else 0.0

    log_probability = (
        gammaln(n + 1)
        - gammaln(x + 1)
        - gammaln(n - x + 1)
        + x * np.log(p)
        + (n - x) * np.log1p(-p)
    )

    return np.exp(log_probability)


# ============================================================
# 2. Région autorisée pour p1
# ============================================================

def p1_bounds(p2, tau, etat):

    if etat == 0:
        # état non différentiel :
        # 1/tau <= p1/p2 <= tau
        low = p2 / tau
        high = min(tau * p2, 1.0)

    elif etat == 1:
        # ESC enrichi :
        # p1/p2 >= tau
        low = tau * p2
        high = 1.0

    elif etat == 2:
        # NPC enrichi :
        # p1/p2 <= 1/tau
        low = 0.0
        high = p2 / tau

    else:
        raise ValueError("Etat doit être 0, 1 ou 2")

    if low >= high:
        return None

    return low, high


# ============================================================
# 3. Probabilité d'une région sous deux lois Beta
# ============================================================

def region_probability(a1, b1, a2, b2, tau, etat):

    def integrand(q):

    # On transforme une probabilité uniforme q
    # en quantile de la loi Beta de p2
        p2 = beta_dist.ppf(q, a2, b2)

        if not np.isfinite(p2):
            return 0.0

        bounds = p1_bounds(p2, tau, etat)

        if bounds is None:
            return 0.0

        low, high = bounds

    # Probabilité que p1 soit dans la région autorisée
        if etat == 1:
        # alpha1 : p1 >= low
            probability_p1 = beta_dist.sf(low, a1, b1)

        elif etat == 2:
        # alpha2 : p1 <= high
            probability_p1 = beta_dist.cdf(high, a1, b1)

        else:
        # alpha0 : low <= p1 <= high
            probability_p1 = (
            beta_dist.cdf(high, a1, b1)
            - beta_dist.cdf(low, a1, b1)
        )

        if not np.isfinite(probability_p1):
            return 0.0

        return probability_p1

    # On évite exactement les deux extrémités
    eps = 1e-10

    result, error = quad(
        integrand,
        eps,
        1.0 - eps,
        epsabs=1e-8,
        epsrel=1e-6,
        limit=200
    )

    return result

# ============================================================
# 4. Probabilité d'émission
# ============================================================

def emission_probability(
    x1,
    x2,
    n1,
    n2,
    a1,
    b1,
    a2,
    b2,
    m,
    tau,
    etat
):

    alpha = 1
    beta_param = m

    # --------------------------------------------------------
    # Log-vraisemblance marginale ESC
    # --------------------------------------------------------

    log_marginal_1 = (
        gammaln(n1 + 1)
        - gammaln(x1 + 1)
        - gammaln(n1 - x1 + 1)
        + betaln(a1, b1)
        - betaln(alpha, beta_param)
    )

    # --------------------------------------------------------
    # Log-vraisemblance marginale NPC
    # --------------------------------------------------------

    log_marginal_2 = (
        gammaln(n2 + 1)
        - gammaln(x2 + 1)
        - gammaln(n2 - x2 + 1)
        + betaln(a2, b2)
        - betaln(alpha, beta_param)
    )

    # --------------------------------------------------------
    # Probabilité de la région sous le posterior
    # --------------------------------------------------------

    posterior_region = region_probability(
        a1,
        b1,
        a2,
        b2,
        tau,
        etat
    )

    # --------------------------------------------------------
    # Probabilité de la même région sous le prior
    # --------------------------------------------------------

    prior_region = region_probability(
        alpha,
        beta_param,
        alpha,
        beta_param,
        tau,
        etat
    )

    if posterior_region <= 0:
        
        return 0.0

    if prior_region <= 0:
        
        return 0.0

    # --------------------------------------------------------
    # Log de la probabilité d'émission
    # --------------------------------------------------------

    log_emission = (
        log_marginal_1
        + log_marginal_2
        + np.log(posterior_region)
        - np.log(prior_region)
    )



    # Protection contre l'underflow
    if log_emission < -745:
        return 0.0

    return np.exp(log_emission)


# ============================================================
# 5. Calcul des émissions pour plusieurs bins
# ============================================================

def calculate_emissions(bins, n1, n2, m, tau):

    emissions = []

    for chromosome, start, x1, x2 in bins:

        a1, b1 = posterior_parameters(
            x1,
            n1,
            m
        )

        a2, b2 = posterior_parameters(
            x2,
            n2,
            m
        )

        e0 = emission_probability(
            x1, x2,
            n1, n2,
            a1, b1,
            a2, b2,
            m, tau,
            0
        )

        e1 = emission_probability(
            x1, x2,
            n1, n2,
            a1, b1,
            a2, b2,
            m, tau,
            1
        )

        e2 = emission_probability(
            x1, x2,
            n1, n2,
            a1, b1,
            a2, b2,
            m, tau,
            2
        )

        emissions.append([
            e0,
            e1,
            e2
        ])

    return np.array(emissions)


# ============================================================
# 6. Lookup table
# ============================================================

def build_emission_lookup(
    candidate_bins,
    n1,
    n2,
    m,
    tau
):

    lookup = {}

    unique_pairs = set()

    for chromosome, start, x1, x2 in candidate_bins:
        unique_pairs.add((x1, x2))

    print(
        "Couples à calculer dans la lookup table :",
        len(unique_pairs)
    )

    for number, (x1, x2) in enumerate(
        sorted(unique_pairs),
        start=1
    ):

        if number % 100 == 0:
            print(
                f"Progression : {number}/{len(unique_pairs)}"
            )

        a1, b1 = posterior_parameters(
            x1,
            n1,
            m
        )

        a2, b2 = posterior_parameters(
            x2,
            n2,
            m
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            e0 = emission_probability(
        x1, x2,
        n1, n2,
        a1, b1,
        a2, b2,
        m, tau,
        0
    )
            if caught_warnings:
                print(
            "WARNING pour le couple :",
            (x1, x2),
            "état α0"
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            e1 = emission_probability(
        x1, x2,
        n1, n2,
        a1, b1,
        a2, b2,
        m, tau,
        1
    )
            if caught_warnings:
                print(
            "WARNING pour le couple :",
            (x1, x2),
            "état α1"
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            e2 = emission_probability(
        x1, x2,
        n1, n2,
        a1, b1,
        a2, b2,
        m, tau,
        2
    )
            if caught_warnings:
                print(
            "WARNING pour le couple :",
            (x1, x2),
            "état α2"
        )

        lookup[(x1, x2)] = (
            e0,
            e1,
            e2
        )

    return lookup


# transformer la table en matrice 
def emissions_from_lookup(candidate_bins, emission_lookup):

    emissions = []

    for chromosome, start, x1, x2 in candidate_bins:

        key = (x1, x2)

        emissions.append(
            emission_lookup[key]
        )

    return np.array(emissions)