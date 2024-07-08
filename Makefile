.PHONY: clean_tr

# Cible pour nettoyer les fichiers .pickle sauf graphe_train.pickle
clean_tr:
	rm -f transformer/*.pickle

clean_lpt:
	rm -f grail-master/data/fb237_v4/train/list_path.pickle