
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
