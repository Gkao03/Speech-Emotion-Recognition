from config import *
from collections import Counter
import random

def preprocess_IEMOCAP(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.emotion != 'xxx']  # only keep data that has emotion label
    # only keep 'neu', 'hap', 'sad', 'ang' labels
    df = df.drop(df[~ ((df.emotion == 'neu') | (df.emotion == 'hap') | (df.emotion == 'sad') | (df.emotion == 'ang'))].index)

    df_unedit = df.copy()
    df_unedit["path"] = df_unedit["path"].apply(lambda x : x.split('/')[-1])
    all_files = list(df_unedit.path)
    file_to_emotion = dict(zip(df_unedit.path, df_unedit.emotion))

    all_full_files = list(df.path)

    return file_to_emotion, all_full_files


def get_unique_emotions_IEMOCAP(file_to_emotion):
    # hardcoded for IEMOCAP
    # emotion_to_label = {'neu': 0, 'fru': 1, 'sad': 2, 'sur': 3, 'ang': 4, 'hap': 5, 'exc': 6, 'fea': 7, 'dis': 8, 'oth': 9}
    emotion_to_label = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
    label_to_emotion = {v: k for k, v in emotion_to_label.items()}

    # counter number of class instances
    emotion_instances_list = [v for v in file_to_emotion.values()]
    counter = Counter(emotion_instances_list)

    file_to_label = {k: emotion_to_label[v] for k, v in file_to_emotion.items()}

    return emotion_to_label, label_to_emotion, file_to_label, counter


def get_train_val_test_IEMOCAP(IEMOCAP_root, all_full_files, file_to_label):
    all_file_paths = [os.path.join(IEMOCAP_root, file_path) for file_path in all_full_files]
    total_instances = len(all_file_paths)

    num_train = round(0.7 * total_instances)
    num_test_all = total_instances - num_train
    num_val = round(0.5 * num_test_all)
    num_test = num_test_all - num_val

    print("number training instances:", str(num_train))
    print("number validation instances:", str(num_val))
    print("number test instances:", str(num_test))
    assert(num_train + num_val + num_test == total_instances)

    shuffled_data_paths = random.sample(all_file_paths, k=total_instances)
    train_list_paths = shuffled_data_paths[:num_train]
    testall_list_paths = shuffled_data_paths[num_train:]
    val_list_paths = testall_list_paths[:num_val]
    test_list_paths = testall_list_paths[num_val:]

    assert(len(train_list_paths) + len(val_list_paths) + len(test_list_paths) == total_instances)

    train_list_labels = [file_to_label[filepath.split('/')[-1]] for filepath in train_list_paths]
    val_list_labels = [file_to_label[filepath.split('/')[-1]] for filepath in val_list_paths]
    test_list_labels = [file_to_label[filepath.split('/')[-1]] for filepath in test_list_paths]

    assert(len(train_list_labels) == len(train_list_paths))
    assert(len(val_list_labels) == len(val_list_paths))
    assert(len(test_list_labels) == len(test_list_paths))

    return train_list_labels, val_list_labels, test_list_labels
