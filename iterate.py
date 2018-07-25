from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

def train():
    # create instance of config
    config = Config()

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                     config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    augment = CoNLLDataset(config.filename_augment, config.processing_word,
                           config.processing_tag, config.max_iter)
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")
    
    # train model
    model.train(train, dev)

    model.reset_graph()
    del model

    evaluate()

    # for i in range(config.niters):
    #     # build model
    #     model = NERModel(config)
    #     model.build()

    #     # train model
    #     model.train(train, dev)
    #     evaluate()

def evaluate():
    # create instance of config
    config = Config()

    # create dataset
    augment = CoNLLDataset(config.filename_augment, config.processing_word,
                           config.processing_tag, config.max_iter)
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    
    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # evaluate
    model.logger.info("\nEvaluation on Test")
    model.evaluate(test)

    augment_pred = []
    model.logger.info("\nEvaluation on Augment")
    model.evaluate(augment, augment_pred)

    print('preds')
    print(augment_pred)

    # Clear memory
    model.reset_graph()
    del model
    
if __name__ == "__main__":
    train()

