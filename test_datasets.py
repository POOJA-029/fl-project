import sklearn
from sklearn.datasets import fetch_openml
import pandas as pd

def test_fetch():
    print("Fetching adult...")
    adult = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    print("Adult complete", adult.frame.shape)
    
    print("Fetching credit-g...")
    credit = fetch_openml(data_id=31, as_frame=True, parser='auto')
    print("Credit complete", credit.frame.shape)

if __name__ == "__main__":
    test_fetch()
