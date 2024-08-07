Brain Tumor Detection
A machine learning project using Resnet-18 for brain tumor detection and Grad-CAM for visualization.

Overview
This project aims to detect brain tumors from MRI images using a Resnet-18 model for classification and Grad-CAM for visual explanations of the predictions.

Table of Contents
Introduction
Installation
Dataset
Model
Training
Evaluation
Visualization
Usage
Contributing
License
Introduction
Brain tumor detection is a critical task in medical diagnosis. This project leverages deep learning and computer vision techniques to accurately identify the presence of tumors in brain MRI images.
Dataset
The dataset used in this project consists of brain MRI images labeled with the presence or absence of tumors. Ensure the dataset is structured as follows:
brain_tumor
│
├── train
│   ├── yes
│   └── no
│
└── val
    ├── yes
    └── no
Model
Architecture
The model architecture is based on Resnet-18, a powerful convolutional neural network commonly used for image classification tasks.

Training
The model was trained using PyTorch on the brain MRI dataset. Training parameters such as learning rate, batch size, and the number of epochs are specified in the notebook.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
The model is trained using the following loop:
num_epochs = 5
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloaders['train'])
    print(f'Epoch [{epoch+1}/{num_epochs}], training Loss: {epoch_loss:.4f}')
    
    scheduler.step()  # Update the learning rate
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(dataloaders['val'])
    accuracy = correct / total * 100

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved")
Evaluation
Evaluate the model using confusion matrix and classification report:
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, dataloaders, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=image_datasets['val'].classes)

    return cm, report

cm, report = evaluate_model(model, dataloaders, device)
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(report)
Visualization
Visualize the model's focus using Grad-CAM:
def grad_cam(model, img_tensor, target_layer):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    def forward_hook(module, input, output):
        model.outputs = output

    def backward_hook(module, grad_in, grad_out):
        model.grads = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    output = output.squeeze(0)

    target_class = output.argmax().item()
    model.zero_grad()
    class_loss = output[target_class]
    class_loss.backward()

    gradients = model.grads.cpu().data.numpy()[0]
    activations = model.outputs.cpu().data.numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)

    handle_forward.remove()
    handle_backward.remove()

    return cam, target_class

def visualize_cam(cam, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img / 255)
    cam_img = cam_img / np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

sample_image_path = r'D:\ml\brain tumor\val\yes\p (30).jpg'

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(sample_image_path)
img_tensor = preprocess(image)

target_layer = model.layer4[1].conv2
cam, target_class = grad_cam(model, img_tensor, target_layer)

visualize_cam(cam, sample_image_path)
Usage
To predict a new image, use the predict_image function:
model.load_state_dict(torch.load('best_model.pth'))

def predict_image(model, image_path, preprocess, target_layer):
    image = Image.open(image_path)
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

    class_names = image_datasets['val'].classes
    prediction = class_names[preds.item()]
    print(f"Predicted: {prediction}")

    cam, target_class = grad_cam(model, img_tensor.squeeze(0), target_layer)
    visualize_cam(cam, image_path)
    return prediction

image_path1 = r"D:\ml\brain tumor new data\train\yes\p (30).jpg"  # yes
image_path2 = r"D:\ml\brain tumor new data\test\no\image(5).jpg"  # no
predict_image(model, image_path1, preprocess, target_layer)
predict_image(model, image_path2, preprocess, target_layer)
Contributing
Contributions are welcome! If you have any improvements or suggestions, feel free to open a pull request or issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

