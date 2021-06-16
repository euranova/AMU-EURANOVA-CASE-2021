"""Script to generate staitstics on the dataset of the third task of ProtestNews2019."""
from sklearn.model_selection import train_test_split

from event_extraction.data_loader.protest_news2019 import ProtestNews2019
from event_extraction.statistic.sequence_dataset import SequenceDataset

def main():
    """Load the data and generate statistics."""
    data_loader = ProtestNews2019('data')
    data_dict = data_loader.load_task3()

    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(data_dict['train_texts'], data_dict['train_labels'], test_size=.2, random_state=13)
    stat = SequenceDataset()
    stat.generate_stats(train_texts, train_labels, 'train_protestnews')



if __name__ == '__main__':
    main()