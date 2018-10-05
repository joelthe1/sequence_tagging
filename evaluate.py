from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

import pickle

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    dev  = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)

    augment = []
    for split in config.augment_list:
        augment.append(
            CoNLLDataset(config.filename_augment.get(split),
                         config.processing_word,
                         config.processing_tag, config.max_iter))

    next_split = min(len(config.augment_list), len(config.splits)-1)
    next_augment = CoNLLDataset(config.filename_augment.get(config.splits[next_split]),
                                config.processing_word,
                                config.processing_tag, config.max_iter)
        

    # evaluate on dev
    model.results_logger.info("\nDev")
    model.evaluate(dev)

    # evaluate on test
    model.results_logger.info("\nTest")
    model.evaluate(test)

    if len(config.augment_list) > 0:
        # evaluate on current augment
        augment_pred, augment_pred_argmax = [], []
        model.results_logger.info("\nAugment split: {}"
                          .format(config.augment_list[-1]))
        model.evaluate(augment[-1], augment_pred, augment_pred_argmax)

        # save current augment predictions
        with open(config.dir_output + 'preds-{}.pkl'
                  .format(config.augment_list[-1]), 'wb') as f:
            pickle.dump(augment_pred, f)

        with open(config.dir_output + 'preds-argmax-{}.pkl'
                  .format(config.augment_list[-1]), 'wb') as f:
            pickle.dump(augment_pred_argmax, f)
            

    # evaluate on the next augment split and save predictions
    augment_pred, augment_pred_argmax = [], []
    model.results_logger.info("\nNext Augment split: {}"
                      .format(config.splits[next_split]))
    model.evaluate(next_augment, augment_pred, augment_pred_argmax)

    # save next augment split predictions
    with open(config.dir_output + 'preds-{}.pkl'.format(config.splits[next_split]), 'wb') as f:
        pickle.dump(augment_pred, f)

    with open(config.dir_output + 'preds-argmax-{}.pkl'.format(config.splits[next_split]), 'wb') as f:
        pickle.dump(augment_pred_argmax, f)
        

if __name__ == "__main__":
    main()
