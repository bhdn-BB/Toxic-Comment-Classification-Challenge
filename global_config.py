import torch

TRAIN_PATH = 'D:\\Toxic-Comment-Classification-Challenge\\kaggle\\working\\train.csv'
MODEL_NAME = "unitary/toxic-bert"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_SIZE = 0.2
TARGET_COL = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_TOKENS = 512
BATCH_SIZE = 64
NUM_LABELS = 6