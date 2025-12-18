import os
import pickle
import numpy as np

INDUSTRY_NPY = r"D:\PythonProject\PDU\data\relation\sector_industry\NASDAQ_industry_relation.npy"
WIKI_NPY = r"D:\PythonProject\PDU\data\relation\wikidata\NASDAQ_wiki_relation.npy"
OUT_PICKLE = r"D:\PythonProject\PDU\data\relation\NASDAQ_File.txt"

def adj_from_relation(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        A = (np.sum(arr, axis=2) != 0).astype(np.int32)
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        A = (arr != 0).astype(np.int32)
    elif arr.ndim == 2:
        M = (arr != 0).astype(np.int32)
        A = ((M @ M.T) > 0).astype(np.int32)
    else:
        raise ValueError(f"Unexpected relation array shape: {arr.shape}")
    np.fill_diagonal(A, 0)
    A = ((A + A.T) > 0).astype(np.int32)
    return A

def to_datagraph_pickle(A: np.ndarray, out_path: str) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square (N,N), got {A.shape}")
    N = A.shape[0]
    graphdata = [np.where(A[i] == 1)[0].tolist() for i in range(N)]
    data_index = list(range(N))
    train_data = [graphdata, data_index]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(train_data, f)
    print(out_path)

if __name__ == "__main__":
    industry = np.load(INDUSTRY_NPY, allow_pickle=True)
    wiki = np.load(WIKI_NPY, allow_pickle=True)
    A_ind = adj_from_relation(industry)
    A_wiki = adj_from_relation(wiki)
    if A_ind.shape != A_wiki.shape:
        raise ValueError(f"Shape mismatch: {A_ind.shape} vs {A_wiki.shape}")
    A = ((A_ind + A_wiki) > 0).astype(np.int32)
    to_datagraph_pickle(A, OUT_PICKLE)
