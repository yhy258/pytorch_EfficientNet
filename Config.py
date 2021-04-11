class config :
    lr = 1e-3
    num_classes = 100
    epochs = 100
    batch_size = 16
    use_ReduceLROnPlateau = True
    efficientnet_num = 6
    se_scale = 4
    save_path = "/content/drive/MyDrive/model_save/my_eff_6.pt"
    stochastic_depth=True
    p = 0.5
    train_loss_history_path = ".pickle"
    val_loss_history_path = ".pickle"
    train_acc_history_path = ".pickle"
    val_acc_history_path = ".pickle"
