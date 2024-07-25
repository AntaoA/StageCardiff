import networkx as nx

# Fonction pour lire le fichier et construire le graphe dirigé
def construire_graphe_de_fichier(nom_fichier):
    G = nx.DiGraph()
    with open(nom_fichier, 'r') as f:
        for ligne in f:
            noeud_source, arete, noeud_destination = ligne.split()
            G.add_edge(noeud_source, noeud_destination, label=arete)
    return G

# Fonction pour supprimer les nœuds avec une somme de degré entrant et sortant inférieure à 3
def supprimer_noeuds_de_faible_degre(G):
    noeuds_a_supprimer = [noeud for noeud in G if G.in_degree(noeud) + G.out_degree(noeud) < 5]
    G.remove_nodes_from(noeuds_a_supprimer)
    return G

# Nom du fichier contenant les arêtes
nom_fichier = "valid.txt"

# Construire le graphe dirigé à partir du fichier
graphe = construire_graphe_de_fichier(nom_fichier)

# Supprimer les nœuds avec une somme de degré entrant et sortant inférieure à 3
for i in range (100):
    graphe = supprimer_noeuds_de_faible_degre(graphe)

# Fonction pour sauvegarder le graphe modifié dans un fichier
def sauvegarder_graphe(G, nom_fichier):
    with open(nom_fichier, 'w') as f:
        for (u, v, data) in G.edges(data=True):
            f.write(f"{u}\t{data['label']}\t{v}\n")


nom_fichier_sortie = "valid+5.txt"

# Sauvegarder le graphe modifié dans un nouveau fichier
sauvegarder_graphe(graphe, nom_fichier_sortie)