
# ==========================
# POSTERIOR
# ==========================

def posterior_parameters(x, n, m):

    alpha = 1
    beta_param = m

    a = alpha + x
    b = beta_param + n - x

    return a, b


def posterior_mean(a, b):

    return a / (a + b)


x_es = 30
x_np = 2

n1 = 1000
n2 = 1000

m = 1000

# Paramètres du posterior
a1, b1 = posterior_parameters(x_es, n1, m)
a2, b2 = posterior_parameters(x_np, n2, m)

# Moyenne du posterior
p_es = posterior_mean(a1, b1)
p_np = posterior_mean(a2, b2)

print("Posterior ESC :", a1, b1)
print("Posterior NPC :", a2, b2)

print("Intensité ESC :", p_es)
print("Intensité NPC :", p_np)
