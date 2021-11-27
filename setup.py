from config import *

# baseline dataset using embeddings
class MyDataset(Dataset):
    def __init__(self, file_list, target_list, recognizer):
        
        self.file_list = file_list
        self.target_list = target_list
        self.num_classes = len(list(set(target_list)))
        self.recognizer = recognizer

        # self.recognizer = read_recognizer()

        # feats, feat_lens = [], []
        # for file in tqdm(file_list):
            
        #     feat = torch.tensor(recognizer.pm.compute(read_audio(file))) # batch, len, features
        #     feat_len = torch.tensor(np.array([feat.shape[0]], dtype=np.int32)) # 1D array
            
        #     feats.append(feat)
        #     feat_lens.append(feat_len)
            

        # feats = pad_sequence(feats,batch_first=True,padding_value=0) # batch,features,len
        # feat_lens = pad_sequence(feat_lens,batch_first=True,padding_value=0).squeeze()
        # idx = torch.argsort(feat_lens,descending=True) # sorting the input in descending order as required by the lstms in AM.
        # self.y = np.array(self.target_list)[idx].tolist()   # reorder
        # tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats[idx], feat_lens[idx]], device_id=-1) # converting to the required tensors

        # # Features
        # output_tensor, input_lengths = recognizer.am(tensor_batch_feat, tensor_batch_feat_len, return_lstm=True) # output_shape: [len,batch,features]
        # assert(len(file_list) == output_tensor.shape[1])

        # self.x = output_tensor
        self.x = file_list
        self.y = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # x = self.x[:, index, :]
        # y = self.y[index]
        # y = torch.Tensor([y])
        # print("inside get item")
        filepath = self.file_list[index]
        x = torch.tensor(self.recognizer.pm.compute(read_audio(filepath)))
        x = x.detach()
        x_len = torch.tensor(np.array([x.shape[0]], dtype=np.int32))
        x_len = x_len.detach()
        y = torch.Tensor([self.target_list[index]])
        return x, x_len, y
