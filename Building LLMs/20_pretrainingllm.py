import importlib
import torch
mod = importlib.import_module("19_evaluatingllmperformance")
gen = importlib.import_module("17_predictnexttoken")
train_loader = mod.train_loader
val_loader = mod.val_loader

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #shorten for the trianing simplicity
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer): #train_loader, and val_loader is the data_loader for that particular texts
    #initialize lists to track loses and tokens seen, 
    #eval_freq = how often to check ?? like how often the quiz happens
    # eval_iter is how much to check each time, means use only 5 batches when measuring the loss
    #optimizer is what updates the weights and AdamW is a popular smart optimizer.Instead of moving every weight by the same amount, it remembers past updates and adjusts each weight’s step size. Weights that have been changing steadily keep a good speed, and jumpy ones get calmed down. That’s why nearly everyone uses it for LLMs.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    #Main training loop 
    for epoch in range(num_epochs): # repeats everything below once per epoch meaning one full pass through the training data
        model.train() # set model to training mode, turns the dropout on 

        for input_batch, target_batch in train_loader: # train_loader is the data_loader
            optimizer.zero_grad() #clears the leftover corrections from the previous batch so they dont pile up 
            loss = mod.calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculate loss gradients 
            optimizer.step() # updates model weights using loss gradients 
            tokens_seen += input_batch.numel() # returns the total number of elements (or tokens)
            global_step +=1

            # optimal evaluation step 
            if global_step % eval_freq == 0: # 5 % 5 would be 0, so after 5 batches do it 
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss) # adds them to the list
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
        #print a sample text after each epoch 
        generate_and_print_sample(model, tokenizer, device, start_context)
        
    return train_losses, val_losses, track_tokens_seen

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    # how many tokens it can handle(256) by checking the size of the position table (256 * 756)
    encoded = mod.text_to_token_ids(start_context, tokenizer).to(device) # turns the start_context which is every effort moves you into token numbers and move them to the CPU or GPU
    with torch.no_grad(): # we are just gnerating text it doesnt need grad
        token_ids = gen.generate_text_sample(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size) # without tracking gradients, since nothing is being learned here, has the model predict 50 more tokens, one at a time
    decoded_text = mod.token_ids_to_text(token_ids, tokenizer) # converts all the tokens even the learned ones back to text
    print(decoded_text.replace("\n", " ")) # compact print format
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # removes dropout
    with torch.no_grad(): # removes the gradient calculation because we are not training but evaluating
        train_loss = mod.calc_loss_loader(train_loader, model, device, num_batches=eval_iter) # this = to eval_iter means it only checks after a few batches instead of a whole dataset, so u can call it often during training without slowing down much 
        val_loss = mod.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# training a GPT model instance for 10 epochs using an AdamW optimizer and the train_model_simple function 

import time
start_time = time.time()

torch.manual_seed(123) # fixes the randomness so u get the same results every run 
model = mod.GPTModel(GPT_CONFIG_124M)
model.to(mod.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # lr is the learning rate how big the nudge is, and weight decay shrinks the weights to help prevent overfitting, 
# AdamW = optimizer that updates the weights after each batch (the "W" = improved weight decay)
# lr=0.0004 → learning rate: how big each weight update step is
# weight_decay=0.1 → every step, gently shrinks all weights toward zero (×0.99996)
#   - useful weights get pushed back up by learning, so they stay
#   - weights only used to memorize small details slowly fade away
#   - keeps the model simpler, so it generalizes to new text instead of memorizing
#   - it's a gentle push: with tiny data (one short story) it can't fully stop overfitting

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, mod.device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=mod.tokenizer)


end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

## 10 epochs leads to overfitting, basically its memorizing the content now, as the content provided to learn from is very small 
# here the validation just gets evaluated and no back propogation does on it.