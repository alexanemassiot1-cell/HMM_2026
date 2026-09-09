import matplotlib.pyplot as plt
import numpy as np

# Nombre de régions DHMS
xu_esc = 3833
xu_npc = 889

my_esc = 6135
my_npc = 4147

categories = ["ESC enrichi", "NPC enrichi"]

xu = [xu_esc, xu_npc]
my = [my_esc, my_npc]

x = np.arange(len(categories))
largeur = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Barres
barres_xu = ax.bar(
    x - largeur / 2,
    xu,
    largeur,
    label="Xu et al.",
    color="violet"
)

barres_nous = ax.bar(
    x + largeur / 2,
    my,
    largeur,
    label="Mon analyse",
    color="pink"
)

# Titres et axes
ax.set_title("Comparaison du nombre de régions DHMS")
ax.set_ylabel("Nombre de régions")
ax.set_xticks(x)
ax.set_xticklabels(categories)

ax.legend()

# Valeurs au-dessus des barres
for barres in [barres_xu, barres_nous]:
    for barre in barres:
        hauteur = barre.get_height()

        ax.text(
            barre.get_x() + barre.get_width() / 2,
            hauteur,
            f"{int(hauteur)}",
            ha="center",
            va="bottom"
        )

plt.tight_layout()

# Sauvegarde
plt.savefig("comparaison_DHMS_Xu.png", dpi=300)

# Affichage
plt.show()