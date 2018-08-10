import os


from .general_utils import get_logger, remove_logger, \
    ensure_path_exists
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs
        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None
        """
        # setup state of execution
        self.set_state()

        # create instance of logger
        self.logger = get_logger(self.path_log)
        self.results_logger = get_logger(self.path_results, logger_name='results')

        # load if requested (default)
        if load:
            self.load()


    def remove_logger(self):
        remove_logger(self.path_log)


    def load(self):
        """Loads vocabulary, processing functions and embeddings
        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)
        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    def set_state(self):
        # 4. get current increment and iteration
        # and create the path
        with open(self.path_state) as f:
            self.curr_increment, self.curr_iter = f.read().strip().split('\n')

        # setup model paths
        self.dir_output = '/lfs1/joel/experiments/sequence_tagging/model/{}/{}/'.format(self.curr_increment, self.curr_iter)
        self.dir_model  = self.dir_output + 'modelweights'
        self.path_log   = self.dir_output + 'log.txt'
        self.path_results = self.dir_output + 'results.txt'

        # list of splits to use in the current run
        # must be subset of splits.
        self.augment_list = []
        self.prev_increment = self.curr_increment
        if self.curr_increment in self.splits:
            self.augment_list = self.splits[:self.splits.index(self.curr_increment) + 1]
            if self.curr_iter == '1':
                self.prev_increment = '0' if self.curr_increment == 'a' else self.augment_list[-2]

            # set the path of last predicted augment split (increment)
            self.path_preds = ''
            prev_iter = sorted(os.listdir('/lfs1/joel/experiments/sequence_tagging/model/{}'.format(self.prev_increment)))[-1]
            self.path_preds = '/lfs1/joel/experiments/sequence_tagging/model/{}/{}/'.format(self.prev_increment, prev_iter)

            # TODO: take the model when incrementing from the best
            # performing previous model based on the dev set
            self.path_prev_model = self.path_preds + 'modelweights'

        # directory for training outputs
        ensure_path_exists(self.dir_output)
        
        
    # general config
    path_state = '/lfs1/joel/experiments/sequence_tagging/state.txt'

    # embeddings
    dim_word = 100
    dim_char = 100

    # glove files
    filename_glove = '/lfs1/shared/embeddings/glove.6B.{}d.txt'.format(dim_word)

    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = 'data/glove.6B.{}d.trimmed.npz'.format(dim_word)
    use_pretrained = True

    # dataset
    # filename_dev = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_dev_1.iobes'
    # filename_test = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_test_1.iobes'
    # filename_train = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_train_1.iobes'
    # filename_augment = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_test_1.iobes'
    # filename_augment_occluded = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_test_1.iobes'
    # filename_augment_40 = '/lfs1/joel/experiments/bigmech/data/bc2gm/temp/bc2gm_dev_1.iobes'

    filename_dev = '/lfs1/joel/experiments/bigmech/data/bc2gm/bc2gm_dev.iobes'
    filename_test = '/lfs1/joel/experiments/bigmech/data/bc2gm/bc2gm_test.iobes'
    filename_train = '/lfs1/joel/experiments/bigmech/data/bc2gm/60-40/60-bc2gm-train.iobes'
    filename_augment_40 = '/lfs1/joel/experiments/bigmech/data/bc2gm/60-40/40-bc2gm-train.iobes'

    # list of all the splits in the augmented data
    splits = ['a', 'b', 'c', 'd']

    filename_augment_10, filename_augment_occluded_10 = {}, {}

    for split in splits:
        filename_augment_10[split] = '/lfs1/joel/experiments/bigmech/data/bc2gm/60-10s/10-bc2gm-train-{}.iobes'.format(split)
        filename_augment_occluded_10[split] = '/lfs1/joel/experiments/bigmech/data/bc2gm/60-10s/10-bc2gm-train-occluded-{}.iobes'.format(split)


    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = 'data/words.txt'
    filename_tags = 'data/tags.txt'
    filename_chars = 'data/chars.txt'

    # training
    train_embeddings = True

    nepochs          = 50
    dropout          = 0.5
    batch_size       = 64
    lr_method        = 'adam'
    lr               = 0.01
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 25
    proba_threshold  = 0.0000002 # None otherwise

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = False # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
