"""Test file for data_loader module."""
from pathlib import Path

from event_extraction.data_loader.protest_news2019 import ProtestNews2019
from event_extraction.data_loader.loader import Loader


def test_extract_from_json():
    """Test function for extract_from_json function of protestnews2019 dataset."""
    doc1 = "Testing sentence to verify the test works."
    doc2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do" + \
        " eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim" + \
        " ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut " + \
        "aliquip ex ea commodo consequat."

    loader = Loader()
    data_path = Path("tests/test_data/train_filled.json")

    train_text, train_label = loader.read_json(data_path)

    assert train_label == [1, 0]
    assert train_text == [doc1, doc2]

def test_loading_with_no_labels():
    """Test if test has no labels the labels are None."""
    data_loader = ProtestNews2019('tests/test_data')

    dict_data = data_loader.load_task3()

    assert dict_data['test_labels'] == None