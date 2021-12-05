from config import *

# baseline dataset using embeddings
class MyDataset(Dataset):
    def __init__(self, file_list, target_list, recognizer):
        
        self.file_list = file_list
        self.target_list = target_list
        self.num_classes = len(list(set(target_list)))
        self.recognizer = recognizer

        self.x = file_list
        self.y = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filepath = self.file_list[index]
        x = torch.tensor(self.recognizer.pm.compute(read_audio(filepath)))
        x = x.detach()
        x_len = torch.tensor(np.array([x.shape[0]], dtype=np.int32))
        x_len = x_len.detach()
        y = torch.Tensor([self.target_list[index]])
        return x, x_len, y

    def pad_collate(self, batch):
        # batch looks like [(x0, xlen0, y0), (x4, xlen4, y4), (x2, xlen2, y2)... ]
        feats = [sample[0] for sample in batch]
        feat_lens = [sample[1] for sample in batch]
        target_list = torch.Tensor([sample[2] for sample in batch])

        feats = pad_sequence(feats, batch_first=True, padding_value=0) # batch, features, len
        feat_lens = pad_sequence(feat_lens, batch_first=True, padding_value=0).squeeze()
        idx = torch.argsort(feat_lens, descending=True) # sorting the input in descending order as required by the lstms in AM.

        targets = target_list[idx]
        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats[idx], feat_lens[idx]], device_id=-1) # converting to the required tensors

        # Features
        output_tensor, input_lengths = self.recognizer.am(tensor_batch_feat, tensor_batch_feat_len, return_lstm=True) # output_shape: [len,batch,features]
        output_tensor = output_tensor.detach()
        input_lengths = input_lengths.detach()
        
        return output_tensor, input_lengths, targets


def get_loader(paths, labels, batch_size=32, num_workers=2, train=True):
    dset = MyDataset(paths, labels)
    args = dict(shuffle=True if train else False, batch_size=batch_size, num_workers=num_workers, collate_fn=dset.pad_collate, drop_last=True)  # change to num_workers=4 on diff platform
    loader = DataLoader(dset, **args)

    return loader
