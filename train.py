from config import *


# specific train/test functions for baseline model
def train_baseline(train_loader, model, optimizer, criterion, device, epoch, train_dset):
    # Train
    model.train()
    avg_loss = 0.0
    
    start_time = time.time()
    train_num_correct = 0
    
    for batch_num, (xbatch, xlen, ybatch) in enumerate(train_loader, 0):
        assert(xbatch.shape[2] == 640)
        optimizer.zero_grad()
        
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)

#         outputs, _ = network(x)  # returns output, embeddings_out
        logits = model(xbatch)  # returns output, embeddings_out_norelu, embeddings_out_relu
#         print("outputs:", outputs.shape)
#         print("argmax of output:", torch.argmax(outputs, axis=1).shape)
#         print("y long:", y.long().shape)
        train_num_correct += (torch.argmax(logits, axis=1) == ybatch).sum().item()

        loss = criterion(logits, ybatch.long())
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        training_loss = avg_loss

        # if batch_num % 5 == 0:
        #     print("5 batches have passed")

        if batch_num % 10 == 9:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch, batch_num + 1, avg_loss / 10))
            training_loss = avg_loss / 10
            avg_loss = 0.0
            
    stop_time = time.time()
    print(f"Training Time {stop_time - start_time} seconds")
    
    train_acc = train_num_correct / len(train_dset)
    print(f"Training Acc: {train_acc}")

    return training_loss, train_acc

# specific train/test functions for ICASSP models
def train_language_model(train_loader_LM, model, opt, criterion, device):

    loss_accum = 0.0
    batch_cnt = 0

    model.train()
    start_time = time.time()
    for batch, (x, lengths, y) in enumerate(train_loader_LM):

        x = x.to(device)
        #lengths = lengths.to(device)
        y = y.long().to(device)
        opt.zero_grad()

        logits = model(x, lengths)
        
        loss = criterion(logits.permute(0,2,1), y)
        loss_score = loss.cpu().item()

        loss_accum += loss_score
        batch_cnt += 1
        loss.backward()
        opt.step()      

    NLL = loss_accum / batch_cnt
        
    return model, NLL


def train_model(train_loader, model, opt, criterion, device, epoch_num):

    loss_accum = 0.0
    batch_cnt = 0

    acc_cnt = 0     #count correct predictions
    err_cnt = 0     #count incorrect predictions

    avg_loss = 0.0

    model.train()
    start_time = time.time()
    for batch, (x, lengths, y) in enumerate(train_loader):
        x = x.to(device)
        #lengths = lengths.to(device)
        y = y.long().to(device)
        opt.zero_grad()

        logits = model(x, lengths)

        loss = criterion(logits, y)
        loss_score = loss.cpu().item()

        avg_loss += loss_score
        training_loss = avg_loss

        loss_accum += loss_score
        batch_cnt += 1
        loss.backward()
        opt.step()

        if batch % 10 == 9:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch_num, batch + 1, avg_loss / 10))
            training_loss = avg_loss / 10
            avg_loss = 0.0

        #model outputs
        out_val, out_indices = torch.max(logits, dim=1)
        tar_indices = y

        for i in range(len(out_indices)):
            if out_indices[i] == tar_indices[i]:
                acc_cnt += 1
            else:
                err_cnt += 1
                     
    training_accuracy =  acc_cnt/(err_cnt+acc_cnt) 
    training_loss = loss_accum / batch_cnt
        
    return model, training_accuracy, training_loss


def test_model(loader, model, opt, criterion, device):
    model.eval()
    acc_cnt = 0
    err_cnt = 0

    for x, lengths, y in loader:
        
        x = x.to(device)
        y = y.long().to(device)
        
        logits = model(x, lengths)

        out_val, out_indices = torch.max(logits, dim=1)
        tar_indices = y

        for i in range(len(out_indices)):
            if out_indices[i] == tar_indices[i]:
                acc_cnt += 1
            else:
                err_cnt += 1

    current_acc = acc_cnt/(err_cnt+acc_cnt)
    
    return current_acc