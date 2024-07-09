import torch
from transformer_param import chemin_t_data, chemin_t, chemin_data_train, device, SEQUENCE_LENGTH, END_TOKEN
import torch.nn.functional as F
import pickle

with open(chemin_t + 'transformer_5-12.pickle', 'rb') as f:
    model = pickle.load(f)
    model.to(device)

with open(chemin_data_train + 'index.pickle', 'rb') as f:
    int_to_rel, rel_to_int, rel_vocab, vocab_input, _, _ = pickle.load(f)

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([rel_to_int[word] for word in words[:SEQUENCE_LENGTH]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, k):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    topk_probs, topk_indices = torch.topk(probabilities, k)
    return topk_probs[0], topk_indices[0]

def text_generator(sentence, generate_length):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions, 1)
        sample += ' ' + int_to_rel[next_token]
        if next_token == rel_to_int[END_TOKEN]:
            break
    print(sample)
    print('\n')

    
def text_generator_with_confidence(sentence, generate_length, k):
    model.eval()
    samples = [(" ".join(sentence.split()), 1.0, [])]        
    for _ in range(generate_length):
        all_candidates = []
        for seq, score, list_prob in samples:
            if seq[-5:] == END_TOKEN:
                all_candidates.append((seq, score, list_prob))
                continue
            int_vector = return_int_vector(seq).to(device)

            if len(int_vector) >= SEQUENCE_LENGTH - 1:
                break

            with torch.no_grad():
                predictions = model(int_vector)
                
            
            topk_probs, topk_indices = sample_next(predictions, k)
            
            
            for i in range(k):
                next_token_index = topk_indices[i].item()
                next_token_prob = topk_probs[i].item()
                next_token = int_to_rel[next_token_index]
                candidate = (seq + " " + next_token, score * next_token_prob, list_prob + [next_token_prob])                                
                all_candidates.append(candidate)
            
        # Order all candidates by their probability scores
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        # Select the k best candidates
        samples = ordered[:k]

    return samples

    
with open(chemin_t_data + "nouvelles_relations.txt", "w") as f:
    j = 0
    for r in vocab_input:
        print(j)
        j += 1
        sentence = r + " <SEP> <START>"
        out = text_generator_with_confidence(sentence, SEQUENCE_LENGTH, 5)
        for i in range(len(out)):
            path, p, pl = out[i]
            f.write(f"{path} : {p} : {pl} \n")


