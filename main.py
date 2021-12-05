from config import *
from setup import *
from preprocess import *
from models import *
from utils import save_model, get_device
from train import train_baseline, validate_baseline

if __name__ == '__main__':
    # hyperparams
    run_num = 1  # change
    batch_size = 32
    lr = 0.001
    weight_decay = 5e-5
    num_epochs = 40
    # in_features = 3 # RGB channels
    momentum = 0.9
    num_classes = 4  # 'neu', 'hap', 'sad', 'ang' hardcoded for IEMOCAP

    recognizer = read_recognizer()  # allosaurus

    IEMOCAP_csv_path = "iemocap_full_dataset.csv"
    file_to_emotion, all_full_files = preprocess_IEMOCAP(csv_path=IEMOCAP_csv_path)
    emotion_to_label, label_to_emotion, file_to_label, counter = get_unique_emotions_IEMOCAP(file_to_emotion)

    IEMOCAP_root = "IEMOCAP_full_release"
    train_list_paths, train_list_labels, val_list_paths, val_list_labels, test_list_paths, test_list_labels = get_train_val_test_IEMOCAP(IEMOCAP_root, all_full_files, file_to_label)

    train_loader, train_dset = get_loader(paths=train_list_paths, labels=train_list_labels, recognizer=recognizer)
    val_loader, val_dset = get_loader(paths=val_list_paths, labels=val_list_labels, recognizer=recognizer)
    test_loader, test_dset = get_loader(paths=test_list_paths, labels=test_list_labels, recognizer=recognizer)

    device = get_device()
    model = BaseLSTM(num_layers=3, num_classes=num_classes, input_size=640, hidden_size=256, dropout=0.1, bidirectional=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    print(model)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5, verbose=True)

    for epoch in num_epochs:
        training_loss, train_acc = train_baseline(train_loader, model, optimizer, criterion, device, epoch, train_dset)
        val_acc = validate_baseline(val_loader, model, scheduler, device, epoch, val_dset)
        
        save_dict = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': training_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'scheduler_state_dict': scheduler.state_dict()
                    }
        save_path = os.path.join("saved_models", f"run{run_num}", f"epoch{epoch}_batchsize{batch_size}_lr{lr}.pth")
        save_model(save_dict=save_dict, save_path=save_path)
