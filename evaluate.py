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

    augment = CoNLLDataset(config.filename_augment_10, config.processing_word,
                           config.processing_tag, config.max_iter)

    augment_next = CoNLLDataset(config.filename_augment_next_10, config.processing_word,
                        config.processing_tag, config.max_iter)

    # evaluate on test
    model.logger.info("\nEvaluation on Test")
    model.evaluate(test)

    # evaluate on current augment
    model.logger.info("\nEvaluation on Augment")
    model.evaluate(augment)

    # evaluate on the next 10% augment and save predictions
    model.logger.info("\nEvaluation on the next 10% Augment")
    augment_pred = []
    model.evaluate(augment_next, augment_pred)

    # save augment predictions
    with open(config.dir_output + 'preds.pkl', 'wb') as f:
        pickle.dump(augment_pred, f)


if __name__ == "__main__":
    main()
