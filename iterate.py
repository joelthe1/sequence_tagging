from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

# create instance of config
config = Config()
        
def train():

    with NERModel(config) as model:
        # create datasets
        dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                             config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word,
                             config.processing_tag, config.max_iter)
        augment = CoNLLDataset(config.filename_augment, config.processing_word,
                               config.processing_tag, config.max_iter)
        augment_occluded = CoNLLDataset(config.filename_augment_occluded,
                                        config.processing_word,
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

    # clear memory
    del model

    # evaluate model
    augment_pred = evaluate()

    for i in range(config.niters):
        with NERModel(config) as model:        
            # create datasets
            dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                                 config.processing_tag, config.max_iter)
            train = CoNLLDataset(config.filename_train, config.processing_word,
                                 config.processing_tag, config.max_iter)
            augment = CoNLLDataset(config.filename_augment, config.processing_word,
                                   config.processing_tag, config.max_iter)
            augment_occluded = CoNLLDataset(config.filename_augment_occluded,
                                            config.processing_word,
                                            config.processing_tag, config.max_iter)
            test  = CoNLLDataset(config.filename_test, config.processing_word,
                                 config.processing_tag, config.max_iter)
            
            # build model
            model = NERModel(config)
            model.build()
            
            model.logger.info('\n\nIteration %s', str(i+1))

            # train model
            # print(augment_pred)
            model.train(train, dev, augment, augment_occluded, augment_pred)
        
        # clear memory
        del model

        augment_pred = evaluate()

def evaluate():
    augment_pred = []
    with NERModel(config) as model:

        # create datasets
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

        model.logger.info("\nEvaluation on Augment")
        model.evaluate(augment, augment_pred)

        # model.logger.debug(augment_pred)

    # clear memory
    del model
    
    return augment_pred
    
if __name__ == "__main__":
    train()

