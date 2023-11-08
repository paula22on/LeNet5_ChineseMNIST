import torch
import torch.nn as nn
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def train_single_epoch(dataloader, model, optimizer, criterion, loss_history, accuracy_history):
    model.train()
    total_accuracy = 0
    total_loss = 0 
    for i, (image, targets) in enumerate(dataloader):
        image, targets = image.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        total_loss += loss.item()

        # Compute metrics
        acc = accuracy(targets, output)
        total_accuracy += acc
        accuracy_history.append(accuracy)
    
    mean_accuracy = total_accuracy / len(dataloader)
    mean_loss = total_loss / len(dataloader)
    accuracy_history.append(mean_accuracy)

    return mean_accuracy, mean_loss




def eval_single_epoch(dataloader, model, optimizer, criterion, loss_history):
    accuracy_history = []
    total_accuracy = 0 
    total_loss = 0 
    model.eval()
    for i, (image, targets) in enumerate(dataloader):
        
        image, targets = image.to(device), targets.to(device)
        output = model(image)
        loss = criterion(output, targets)
        total_loss += loss.item()

        # Compute accuracy if needed
        acc = accuracy(targets, output)
        total_accuracy += acc

    # If you want to compute accuracy, do it similar to the training loop
    mean_accuracy = total_accuracy / len(dataloader)
    mean_loss = total_loss / len(dataloader)
    accuracy_history.append(mean_accuracy)

    return mean_accuracy, mean_loss


def train_model(config):

    # Apply data transform
    data_transform = transforms.Compose([
                     transforms.Resize((32, 32)),  # Resize the image to a specific size
                     transforms.ToTensor(),         # Convert the image to a tensor
                     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
                ])


    my_dataset = MyDataset(images_path='datasets/data',
                           labels_path='datasets/chinese_mnist.csv',
                           transform=data_transform
                           )
    my_model = MyModel(num_classes=15).to(device)


    lengths = [10000, 2500, 2500]
    train_data, eval_data, test_data = torch.utils.data.random_split(my_dataset, lengths)

    dataloader_train = DataLoader(train_data, batch_size=64, shuffle = True)
    dataloader_eval = DataLoader(eval_data, batch_size=64, shuffle = True)
    dataloader_test = DataLoader(test_data, batch_size=64, shuffle = True)

    #Setting the loss function
    criterion = nn.CrossEntropyLoss()

    #Setting the optimizer with the model parameters and learning rate
    learning_rate = 0.01
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    #this is defined to print how many steps are remaining when training
    total_step = len(dataloader_train)

    loss_history = []
    accuracy_history = []

    for epoch in range(config["epochs"]):
        train_acc, train_loss = train_single_epoch(dataloader_train, my_model, optimizer, criterion, loss_history, accuracy_history)
        eval_acc, eval_loss = eval_single_epoch(dataloader_eval, my_model, optimizer, criterion, loss_history)
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Eval Accuracy: {eval_acc:.4f}, Eval Loss: {eval_loss:.4f}")


    return my_model


if __name__ == "__main__":

    config = {
        "epochs": 5,
        "hyperparam_2": 2,
    }
    train_model(config)
