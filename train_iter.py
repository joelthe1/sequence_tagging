from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

import pickle


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    augment_occluded, augment_preds = [], []
    for split in config.augment_list:
        augment_occluded.append(CoNLLDataset(
            config.filename_augment_occluded_10.get(split),
                                    config.processing_word,
                                    config.processing_tag, config.max_iter))

        with open(config.dir_output + 'preds-{}.pkl'.format(split), 'rb') as f:
            augment_preds.append(pickle.load(f))
            if len(augment_preds[-1]) == 0:
                raise AttributeError('Error while trying to \
                load augment predictions from pickle.')

    # print(len(augment_preds))
    
    # train model
    model.train(train, dev, augment_occluded, augment_preds)


if __name__ == "__main__":
    main()
