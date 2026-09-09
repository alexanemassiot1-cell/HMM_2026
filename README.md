# Implémentation de ChIPDiff – HMM pour la détection de modifications différentielles des histones

## Description

Ce projet consiste à implémenter en Python l'approche **ChIPDiff** décrite par Xu et al. (2008), basée sur un **modèle de Markov caché (HMM)** pour identifier des sites de modification différentielle des histones à partir de données ChIP-seq.

L'implémentation est appliquée à la modification histonique **H3K27me3** dans deux types cellulaires :

- **ESC** : cellules souches embryonnaires (*Embryonic Stem Cells*)
- **NPC** : cellules progénitrices neurales (*Neural Progenitor Cells*)

L'objectif est de comparer les niveaux de H3K27me3 entre les deux conditions et d'identifier les régions différentiellement modifiées (DHMS).

---

## Organisation du projet

Le projet contient principalement les fichiers suivants :

```text
HMM_2026/
│
├── main.py
├── count_reads.py
├── emission.py
├── baum_welch.py
├── etat_transission.py
├── analyse_stat.py
│
├── files/
│   ├── GSM307619_ES.H3K27me3.aligned.txt.gz
│   └── GSM307614_NP.H3K27me3.aligned.txt.gz
│
├── resultats_HMM.tsv
└── regions_DHMS.tsv