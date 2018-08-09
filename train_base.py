from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

from subprocess import run

def main():
    # create instance of config
    config = Config()

    # clean-up any previous predictions
    # run('rm {}preds-*.pkl'.format(config.dir_output), shell=True)

    # build model
    model = NERModel(config)
    model.build()

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
