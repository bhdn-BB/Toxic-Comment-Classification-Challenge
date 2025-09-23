import torch
from torch.utils.data import Dataset as BaseDataset
from global_config import NUM_TOKENS, TARGET_COL


class ToxicDataset(BaseDataset):
    def __init__(
            self,
            tokenizer,
            df,
            max_tokens=NUM_TOKENS,
    ) -> None:
        self.tokenizer = tokenizer
        self.texts = df["comment_text"].values
        self.labels = df[TARGET_COL].values
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        x = {key: torch.tensor(val) for key, val in encoding.items()}
        y = torch.from_numpy(self.labels[index]).float()
        return x, y