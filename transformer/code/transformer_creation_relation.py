from transformer_train import *



    
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

    
generate_length = 8


with open("nouvelles_relations.txt", "w") as f:
    for r in rel_vocab:
        if r[-6:] == "_input":
            sentence = r + " <SEP> <START>"
            out = text_generator_with_confidence(sentence, generate_length, 5)
            for i in range(len(out)):
                path, p, pl = out[i]
                f.write(f"{path} : {p} : {pl} \n")

    
