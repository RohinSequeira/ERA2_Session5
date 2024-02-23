from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def transform_train_data():
  """
  Download train and test data under 'train_loader' and 'test_loader' respectively.
  Data is converted to Tensors while Normalizing the values between 0 and 1
  """
  # Train data transformations
  train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
      transforms.Resize((28, 28)),
      transforms.RandomRotation((-15., 15.), fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
      ])

  return train_transforms

def transform_test_data():
  # Test data transformations
  test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
  return test_transforms

def get_train_data():
  #Download and return train data
  train_data = datasets.MNIST('../data', train=True, download=True, transform=transform_train_data())
  return train_data

def get_test_data():
  #Download and return test data
  test_data = datasets.MNIST('../data', train=False, download=True, transform=transform_test_data())
  return test_data


def visualize_train_test_loss_acc(train_losses, train_acc, test_losses, test_acc):
  #Visualize our losses and accuracy
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")


def visualize_sample_data(train_loader):
  #Visualize sample images from the training data set.
  batch_data, batch_label = next(iter(train_loader))

  fig = plt.figure()

  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

