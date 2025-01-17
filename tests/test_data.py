import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rice_images.data import load_data

def test_my_dataset():
    """Test the MyDataset class."""
    train_dataset, val_dataset, test_dataset = load_data()
    assert len(train_dataset)==int(0.7 * 75000)
    assert len(val_dataset)==int(0.15 * 75000)
    assert (len(train_dataset)+len(val_dataset)+len(test_dataset))==75000

if __name__ =="__main__":
    # testing the data doesn't work without data.
    # Data needs to be stored somewhere else so that this test can be run by github. Until then, we will assert true
    # test_my_dataset()
    assert True
