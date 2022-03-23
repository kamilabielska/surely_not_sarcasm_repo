import numpy as np
import time
import datetime
import copy
import torch


def train_model(model, criterion, optimizer, scheduler, epochs, dataloaders, data_size, device, es_patience=None, rep=100):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    epochs_no_improvement = 0

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print('')
        print(fr'======== Epoch {epoch + 1} / {epochs} ========')
        print('Training...')
        t0 = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            dataloader = dataloaders[phase]
            running_loss = 0
            running_corrects = 0

            for step, batch in enumerate(dataloader):
                if phase == 'train' and step % rep == 0 and not step == 0:
                    print(fr'Batch {step:>5,} of {len(dataloader):>5,}. Elapsed: {(time.time()-t0):.2f} s.')

                ids = batch[0].to(device)
                masks = batch[1].to(device)
                labels = batch[2].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids, attention_mask=masks, return_dict=False)
                    _, preds = torch.max(outputs[0], 1)
                    loss = criterion(outputs[0], labels)
            
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*ids.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/data_size[phase]
            epoch_accuracy = running_corrects.double()/data_size[phase]
            
            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_accuracy'].append(epoch_accuracy)

            print(fr'== {phase} == loss: {epoch_loss:.4f} accuracy: {epoch_accuracy:.4f}')

            if phase == 'val':
                if epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
        
        if es_patience is not None and epochs_no_improvement >= es_patience:
            print('')
            print(fr'No validation accuracy improvement for {es_patience} consecutive epochs. Stopping training...')
            break
    
    print('')
    print(fr'Training finished. Best validation accuracy: {best_acc:.4f}.')
    model.load_state_dict(best_model_wts)
    return model, history

def test_model(model, dataloader, data_size, device):
    model.eval()
    
    pred_labels, test_labels = [], []
    running_corrects = 0
    
    for batch_idx, (ids, masks, labels) in enumerate(dataloader):

        ids, labels, masks = [x.to(device) for x in [ids, labels, masks]] 
        
        with torch.no_grad():
            outputs = model(ids, attention_mask=masks, return_dict=False)
            _, preds = torch.max(outputs[0], 1)

        pred_labels += preds.tolist()
        test_labels += labels.data.tolist()

        running_corrects += torch.sum(preds == labels.data)
        
    accuracy = running_corrects.double()/data_size
    print(fr'test accuracy: {accuracy:.4f}')
    
    return np.array(pred_labels), np.array(test_labels)