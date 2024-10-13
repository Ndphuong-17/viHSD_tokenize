import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import torch.optim as optim


class MultiTaskModel(nn.Module):
    def __init__(self, embedding_model, classification_model, dropout_rate=0.1):
        super(MultiTaskModel, self).__init__()
        self.embedding_model = embedding_model
        self.classification_head = classification_model
        self.dropout = nn.Dropout(dropout_rate)  # Initialize dropout layer

    def forward(self, input_ids, attention_mask):
        """
        input_ids: Tensor of shape (batch_size, max_length)
        attention_mask: Tensor of shape (batch_size, max_length)
        """
        # Get token-level embeddings from the embedding model
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, max_length, hidden_dim)

        # Apply mean pooling over the last hidden state
        pooled_output = last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_dim)

        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)

        # Pass the pooled embeddings through the classification head
        # Assuming that `classification_head` requires input_ids and attention_mask
        logits = self.classification_head(input_ids=input_ids, attention_mask=attention_mask).logits 

        return logits





def setup_model(model_class, embedding_model, classification_model, lr=5e-6, weight_decay=1e-5, num_epochs=2):
    """
    Sets up the multi-task model, criterion, optimizer, and device for training and testing.
    
    Args:
        embedding_model: The base embedding model.
        classification_model_class: The class of the multi-task classification model.
        lr (float): Learning rate for the optimizer. Default is 5e-6.
        weight_decay (float): Weight decay for the optimizer. Default is 1e-5.
        num_epochs (int): Number of epochs for training. Default is 2.

    Returns:
        model (torch.nn.Module): The instantiated multi-task model.
        criterion (nn.Module): The loss function for the model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        device (torch.device): The device to be used (CPU or GPU).
        num_epochs (int): The number of training epochs.
    """
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create an instance of the multi-task model and move it to the device
    model = model_class(embedding_model, classification_model).to(device)
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()  # Assuming a multi-class classification task
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Return the setup components
    return model, criterion, optimizer, device, num_epochs

def train(model, train_dataloader, dev_dataloader, criterion, optimizer, device, num_epochs):
    """
    Train the model with specified parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (DataLoader): Dataloader for training data.
        dev_dataloader (DataLoader): Dataloader for validation data.
        criterion (torch.nn.Module): Loss function for binary classification.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to perform training on.
        num_epochs (int): Number of training epochs.
    """
    model.to(device)  # Ensure model is on the correct device

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode at the start of each epoch
        total_loss = 0
        print(f'Epoch: {epoch + 1}')

        # Training loop
        for data in tqdm(train_dataloader, desc='Training', leave=False):
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device, dtype=torch.float)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.squeeze(), labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation loop
        model.eval()  # Set model to evaluation mode for validation
        val_loss = 0
        preds, targets = [], []

        with torch.no_grad():
            for data in tqdm(dev_dataloader, desc='Validation', leave=False):
                input_ids = data['input_ids'].squeeze(1).to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['label'].to(device, dtype=torch.float)

                # Forward pass for validation
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.squeeze(), labels)
                val_loss += loss.item()

                # Collect predictions and targets for F1 score calculation
                preds.append(logits.squeeze().cpu())
                targets.append(labels.cpu())

        # Concatenate predictions and targets
        preds = torch.cat(preds).numpy().flatten()
        targets = torch.cat(targets).numpy().flatten()

        # Convert logits to binary predictions using thresholding
        preds = (preds > 0.5).astype(int)

        # Calculate macro F1-score
        f1 = f1_score(targets, preds, average='macro')

        # Display training and validation losses and metrics
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(dev_dataloader)
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Macro F1-Score: {f1:.4f}')


def test(model, test_dataloader, device):
    """
    Evaluate the model on the test dataset and compute the macro F1 score.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        preds (np.ndarray): The predicted labels.
        targets (np.ndarray): The actual target labels.
    """
    model.eval()
    preds, targets = [], []

    # Iterate through the test dataloader
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc='Testing'):
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device, dtype=torch.float)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Store predictions and targets
            preds.append(logits.squeeze().cpu())
            targets.append(labels.cpu())

    # Concatenate the predictions and targets
    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    # Apply thresholding to convert logits into binary predictions
    preds = (preds > 0.5).astype(int)

    # Calculate macro F1 score
    f1 = f1_score(targets, preds, average='macro')
    print(f"Macro F1 Score: {f1:.4f}")

    return preds, targets
