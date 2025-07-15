from datasets import load_dataset
from tokenizer import tokenizer, PAD_ID, SOS_ID, EOS_ID, sp
import torch
from transformer_model import transformer
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

def padding(index_list, max_len):
    if len(index_list) < max_len:
        index_list += (max_len - len(index_list)) * [PAD_ID]
    else:
        index_list = index_list[:max_len]
    return index_list

def batch(batch_data):
    encoder_tensors = []
    decoder_tensors = []
    target_tensors = []
    for data in batch_data:
        context = data["previous_utterance"]
        if isinstance(context, list):
            context = " ".join(context)
        free_msgs = data["free_messages"]
        if isinstance(free_msgs, list):
            if len(free_msgs) == 0:
                continue
            last_msg = free_msgs[-1]
            if isinstance(last_msg, dict):
                reply = last_msg.get("text", "")
            else:
                reply = last_msg
        else:
            reply = free_msgs

        if len(reply.strip()) == 0:
            continue

        encoder_index = tokenizer(context)
        target_index = tokenizer(reply) + [EOS_ID]
        decoder_index = [SOS_ID] + target_index[:-1]

        padding_encoder = padding(encoder_index, 100)
        padding_target = padding(target_index, 100)
        padding_decoder = padding(decoder_index, 100)

        encoder_tensors.append(padding_encoder)
        decoder_tensors.append(padding_decoder)
        target_tensors.append(padding_target)

    if len(encoder_tensors) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    encoder_tensor = torch.tensor(encoder_tensors, dtype=torch.long)
    decoder_tensor = torch.tensor(decoder_tensors, dtype=torch.long)
    target_tensor = torch.tensor(target_tensors, dtype=torch.long)

    return encoder_tensor, decoder_tensor, target_tensor


if __name__ == "__main__":

    dataset = load_dataset("blended_skill_talk", trust_remote_code=True, cache_dir="./data/hf_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = sp.get_piece_size()
    model = transformer(vocab_size, embedding_dim=256)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2, 
    verbose=True)

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=0.1 
    )
    epochs = 30
    save_path = "./transformer.pth"
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("find exist model")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=16,
        shuffle=True,
        collate_fn=batch
    )

    patience = 3
    epochs_no_improve = 0
    early_stop = False

    accuracy_path = os.path.join('best_accuracy.pkl')
    if os.path.exists(accuracy_path):
        with open(accuracy_path, 'rb') as f:
            best_accuracy = pickle.load(f)
        print(f"Loaded previous best_accuracy: {best_accuracy:.2f}%")
    else:
        best_accuracy = 0
        print("No previous best_accuracy found, starting from 0%.")

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=16,
        shuffle=False,
        collate_fn=batch
    )

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        train_correct_tokens = 0
        train_total_tokens = 0
        for encoder_tensor, decoder_tensor, target_tensor in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            encoder_tensor = encoder_tensor.to(device)
            decoder_tensor = decoder_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output = model(encoder_tensor,decoder_tensor)

            batch_size = output.shape[0]
            token_size = output.shape[1]
            vocab_size = output.shape[2]

            output_flat = output.view(batch_size*token_size, vocab_size)
            target_flat = target_tensor.view(batch_size * token_size)

            loss = loss_fn(output_flat, target_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = output_flat.argmax(dim=1)
            mask = target_flat != PAD_ID
            train_correct_tokens += (preds == target_flat).masked_select(mask).sum().item()
            train_total_tokens += mask.sum().item()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        train_accuracy = 100 * train_correct_tokens / train_total_tokens
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        val_correct_tokens = 0
        val_total_tokens = 0
        val_loss = 0

        with torch.no_grad():
            for encoder_tensor, decoder_tensor, target_tensor in val_loader:
                encoder_tensor = encoder_tensor.to(device)
                decoder_tensor = decoder_tensor.to(device)
                target_tensor = target_tensor.to(device)

                output = model(encoder_tensor, decoder_tensor)

                batch_size, seq_len, vocab_size = output.shape
                output_flat = output.view(batch_size * seq_len, vocab_size)
                target_flat = target_tensor.view(batch_size * seq_len)

                val_loss += loss_fn(output_flat, target_flat).item()

                preds = output_flat.argmax(dim=1)
                mask = target_flat != PAD_ID
                val_correct_tokens += (preds == target_flat).masked_select(mask).sum().item()
                val_total_tokens += mask.sum().item()
        val_accuracy = 100 * val_correct_tokens / val_total_tokens
        scheduler.step(val_accuracy)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        model.train()

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with val_accuracy: {best_accuracy:.2f}%")
            epochs_no_improve = 0

            with open(accuracy_path, 'wb') as f:
                pickle.dump(best_accuracy, f)

        else:
            epochs_no_improve+=1
            if epochs_no_improve >= patience:
                early_stop = True
        if early_stop:
            print("early stop triggered")
            break

    torch.save(model.state_dict(),save_path)
    print("finished")