import torch
import torch.nn as nn
import torch.optim as optim


def run(model, train_data, train_label, test_data, test_label, num_epochs = 300, batch_size = 80, print_freq = 20):
    # Record list
    loss_list = []
    acc_train_list = []
    acc_test_list = []
    
    # Move data do gpu
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data  = train_data.to(device)
    train_label = train_label.to(device)
    test_data   = test_data.to(device)
    test_label  = test_label.to(device)
    
    # Setup loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(1,num_epochs+1):
        # Loss in each epoch
        epoch_loss = 0.0

        # Make train_data run in batch
        permutation = torch.randperm(train_data.shape[0])

        # Run with data batches
        for i in range(0,train_data.shape[0], batch_size):
            # Zero the gradients
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_data[indices], train_label[indices]

            # Run the net & Update
            outputs = model(batch_x)
            #print('output = ',outputs)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calclate accuracy
        acc_train_list.append(model.infer_and_cal_acc(train_data,train_label))
        acc_test_list.append(model.infer_and_cal_acc(test_data,test_label))


        loss_list.append(epoch_loss)
        if epoch % print_freq == 0:
            print('epoch ',epoch,' : loss = ',epoch_loss)
            
    return loss_list, acc_train_list, acc_test_list