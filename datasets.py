import numpy as np

def build_reversing_dataset(
    num_train=1000,
    num_test=100,
    num_ints=10,
    seq_length=10,
    random_seed=0):
    
    dataset = {}

    np.random.seed(random_seed)
    for subset in ["train", "test"]:
        dataset[subset] = {}
        dataset[subset]["inputs"] = np.random.randint(
            num_ints, 
            size=[num_train if subset == "train" else num_test, seq_length],
            dtype=np.int32) 
        dataset[subset]["outputs"] = dataset[subset]["inputs"][:, ::-1]
    return dataset

if __name__ == "__main__":
    print(build_reversing_dataset(num_train=5, num_test=5))
