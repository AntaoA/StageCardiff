.PHONY: clean

# Cible pour nettoyer les fichiers .pickle sauf graphe_train.pickle
clean:
	find . -name "*.pickle" -type f ! -name "graphe_train.pickle" -exec rm {} +
