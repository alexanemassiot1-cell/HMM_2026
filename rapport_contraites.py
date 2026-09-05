ratio = p_es / p_np
tau = 3 # seuil

if ratio >= 1/tau and ratio <= tau:
    state = 0       # α0

elif ratio > tau:
    state = 1       # α1 : ESC enrichie

else:
    state = 2       # α2 : NPC enrichie