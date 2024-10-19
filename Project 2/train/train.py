from data import create_dataloaders
from models import get_model
import matplotlib as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import os  # To handle file paths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Using CPU")

def train_and_validate(model_name, num_epochs=30, learning_rate=0.001, batch_size=32, optimizer_type='adam', accuracy_threshold=99.0):
    # Initialize model and move to GPU
    num_classes = 8
    model = get_model(model_name, num_classes).to(device)
    
    # Create dataloaders with GPU optimization
    train_loader, val_loader = create_dataloaders('./dataset', batch_size)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress tracking
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision training
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        
        # Check if validation accuracy threshold is reached
        if val_accuracy >= accuracy_threshold:
            print(f"\nReached {accuracy_threshold}% validation accuracy at epoch {epoch+1}")
            print("Stopping training early...")
            break
    
    # Create directory for saving charts if not exists
    chart_dir = './charts'
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)
    
    # Plot and save loss curves
    plt.figure(figsize=(14, 7))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.title(f"Loss Curves for {model_name}")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.title(f"Accuracy Curves for {model_name}")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plots as images
    chart_filename = f"{model_name}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_opt{optimizer_type}_charts.png"
    plt.savefig(os.path.join(chart_dir, chart_filename))
    
    print(f"Charts saved: {os.path.join(chart_dir, chart_filename)}")
    
    # Display the charts
    plt.show()
    
    return model

model_names = ['resnet18', 'vgg16', 'efficientnet_b0']

# Define hyperparameter grids
learning_rates = [0.001, 0.0001]
batch_size = 256
optimizer = 'adam'
num_epochs = 10  # Number of epochs to try

# Variables to track the best model overall
best_model_overall = None
best_accuracy_overall = 0
best_model_details = None
                    
for model_name in model_names:
    for learning_rate in learning_rates:
        trained_model = train_and_validate(
            model_name=model_name,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer
        )
        
        scripted_model = torch.jit.script(trained_model)

        # Save the TorchScript model
        model_filename = f"{model_name}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}_opt{optimizer}_scripted.pt"
        scripted_model.save(model_filename)
        print(f"TorchScript model saved: {model_filename}")
        
# Print the details of the best model
print(f"\nBest Model Details:\n{best_model_details}\nBest Validation Accuracy: {best_accuracy_overall:.2f}%")