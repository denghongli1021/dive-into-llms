from torch.utils.data import Dataset
import torch
import pandas as pd

def load_data(args, split):
    df = pd.read_csv(f"{args.data_root}/{split}.csv")
    texts = df['text'].values.tolist()
    labels = df['target'].values.tolist()
    return texts, labels

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data[0]
        self.labels = data[1]
        self.is_test = is_test
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        text = str(self.texts[index]) # 取出文字並字串化
        # Tokenizer 編碼 (batch_encode_plus)：
        # max_length=self.max_length：限制最大長度。
        # padding="max_length"：不足最大長度時填充 Padding token（如 [PAD]）。
        # truncation=True：超過最大長度時自動截斷。
        # return_tensors="pt"：直接回傳 PyTorch Tensors。
        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            # pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        data_sample = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }
        if not self.is_test:
            label = self.labels[index]
            target_ids = torch.tensor(label).squeeze()
            data_sample["labels"] = target_ids 
        return data_sample
            