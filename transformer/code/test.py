import pickle

with open("transformer/data/tableau_model.pickle", "rb") as f:
    tab_t, prob, prob_ft = pickle.load(f)
    
diff = [0.0] * len(tab_t)
for i in range(len(tab_t)):
    diff[i] = prob_ft[i] - prob[i]
    print(f"{i} {diff[i]}")

print("ok")
