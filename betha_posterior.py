
def posterior_mean(x, n, m):
    alpha = 1
    beta_param = m

    return (
        alpha + x
    ) / (
        alpha + beta_param + n
    )

x1 = 10
n1 = 1000
m = 10000
n2 = 1000


x_es = 50
n_es = n1


p_es = posterior_mean(
    x_es,
    n_es,
    m
)

print("Intensité ESC :", p_es)

x_np = 10
n_np = n2

p_np = posterior_mean(
    x_np,
    n_np,
    m
)

print("Intensité NPC :", p_np)