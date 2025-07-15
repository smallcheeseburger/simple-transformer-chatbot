import os
from transformer_model import transformer
import torch
from train import padding
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

PAD_ID = sp.piece_to_id('<pad>')
UNK_ID = sp.piece_to_id('<unk>')
SOS_ID = sp.piece_to_id('<sos>')
EOS_ID = sp.piece_to_id('<eos>')

model = transformer(sp.get_piece_size(), embedding_dim=256)
save_path = "./transformer.pth"
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print("find exist model")
model = model.to(device)
model.eval()

input_sentence = "Hello world"
input_index = sp.encode(input_sentence, out_type=int)
padding_input = padding(input_index, 100)

tensor_input = torch.tensor(padding_input, dtype=torch.long)
encoder_tensor = tensor_input.unsqueeze(0).to(device)

generated = [SOS_ID]

for step in range(100):
    tokens = generated + [PAD_ID] * (100 - len(generated))
    decoder_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(encoder_tensor, decoder_tensor)

    temperature = 0.8
    top_k = 20

    next_token_logits = output[0, len(generated) - 1, :]
    next_token_logits = next_token_logits / temperature
    values, indices = torch.topk(next_token_logits, top_k)
    probs = torch.softmax(values, dim=-1)
    next_token_id = indices[torch.multinomial(probs, 1)].item()

    if next_token_id == EOS_ID:
        break

    generated.append(next_token_id)

predicted_tokens = sp.decode(generated[1:])
print("Predicted tokens:", predicted_tokens)