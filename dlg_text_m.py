import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def deep_leakage_from_gradients(model, gradients, dummy_inputs, dummy_labels, learning_rate=1.0, max_iters=30):
    optimizer = optim.LBFGS([dummy_inputs, dummy_labels], lr=learning_rate)

    for iter in range(max_iters):
        def closure():
            optimizer.zero_grad()
            dummy_outputs = model(dummy_inputs)[0]
            dummy_loss = nn.MSELoss()(dummy_outputs, dummy_labels)
            dummy_loss.backward()
            dummy_gradients = [p.grad for p in model.parameters()]
            grad_distance = sum((dg - g).norm()**2 for dg, g in zip(dummy_gradients, gradients))
            grad_distance.backward()
            return grad_distance

        optimizer.step(closure)
        print(f"Iteration {iter+1}/{max_iters} completed.")

    return dummy_inputs, dummy_labels


def prepare_dummy_data(tokenizer, sequence_length):
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, sequence_length))
    dummy_input_ids[0, 0] = tokenizer.cls_token_id
    dummy_input_ids[0, -1] = tokenizer.sep_token_id
    dummy_labels = torch.randn_like(dummy_input_ids, dtype=torch.float)
    return dummy_input_ids, dummy_labels


def reconstruct_text(model, tokenizer, gradients, sequence_length=20):
    dummy_input_ids, dummy_labels = prepare_dummy_data(tokenizer, sequence_length)
    dummy_input_ids.requires_grad = True
    dummy_labels.requires_grad = True

    reconstructed_input_ids, reconstructed_labels = deep_leakage_from_gradients(model, gradients, dummy_input_ids, dummy_labels)

    reconstructed_text = tokenizer.decode(reconstructed_input_ids[0].detach().cpu().numpy(), skip_special_tokens=True)
    return reconstructed_text


# Assume you have the gradients from the real training data
real_gradients = [torch.randn_like(p) for p in model.parameters()]

reconstructed_text = reconstruct_text(model, tokenizer, real_gradients)
print("Reconstructed Text:", reconstructed_text)
