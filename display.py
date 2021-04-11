import pickle
import matplotlib.pyplot as plt
from Config import config
with open(config.train_loss_history_path, 'rb') as f:
    train_history = pickle.load(f)
with open(config.val_loss_history_path, 'rb') as f:
    val_history = pickle.load(f)

with open(config.train_acc_history_path, 'rb') as f:
    train_acc_history = pickle.load(f)
with open(config.val_acc_history_path, 'rb') as f:
    val_acc_history = pickle.load(f)

plt.title('Train-Validation Loss')
plt.plot(range(1, config.epochs +1), train_history, label='train')
plt.plot(range(1, config.epochs +1), val_history, label='val')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()


plt.title('Train-Validation Accuracy')
plt.plot(range(1, config.epochs +1), train_acc_history, label='train')
plt.plot(range(1, config.epochs +1), val_acc_history, label='val')
plt.ylabel('Accuracy')
plt.xlabel('Training Epochs')
plt.legend()
plt.show()