import numpy as np
import pandas as pd
from pathlib import Path

#setting random seed
np.random.seed(42)

#loading the data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "train.csv"
train = pd.read_csv(train_path)
print(train.head(2))

test_path = DATA_DIR / "test.csv"
test = pd.read_csv(test_path)
print(test.head(2))