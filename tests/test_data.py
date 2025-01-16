from rice_images.data import load_data, pre_process_data

def test_dataset_size(train_dataset, val_dataset, test_dataset):
    """Test the MyDataset class."""
    assert len(train_dataset)==int(0.7 * 75000)
    assert len(val_dataset)==int(0.15 * 75000)
    assert (len(train_dataset)+len(val_dataset)+len(test_dataset))==75000


def test_normalization(train_dataset, val_dataset, test_dataset):
    """Test the MyDataset class."""
    assert train_dataset[0][0].max()<=1
    assert train_dataset[0][0].min()>=0
    assert val_dataset[0][0].max()<=1
    assert val_dataset[0][0].min()>=0
    assert test_dataset[0][0].max()<=1
    assert test_dataset[0][0].min()>=0


if __name__ =="__main__":
    # testing the data doesn't work without data. 
    # Data needs to be stored somewhere else so that this test can be run by github. Until then, we will assert true
    # Pre-process the data
    pre_process_data()

    # Load the data
    train_dataset, val_dataset, test_dataset = load_data()
    
    # Perform the data tests
    # test_dataset_size() 
    # test_normalization()
    assert True

