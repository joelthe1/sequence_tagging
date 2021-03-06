import os
import time
import sys
import logging
import numpy as np


def get_logger(filename, logger_name='logger'):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    # create and set log-level of logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # create handler and set handler for stderr
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    # create handler for logging to file
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    
    if logger_name == 'logger':
        handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger(logger_name).addHandler(handler)

    return logger


def remove_logger(filename):
    handler = logging.FileHandler(filename)
    logging.getLogger().removeHandler(handler)


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


'''

Dev
|69.49|65.88|73.52

Test
|69.07|66.17|72.24

Augment split: a
|82.59|78.93|86.62

Next Augment split: b
|68.70|65.16|72.65
'''

def get_best_model_iter(path_increment):
    model_scores = {'test': [],
                    'dev': [],
                    'augm': [],
                    'next_augm': []}

    for root, dirs, files in os.walk(path_increment):
        for name in files:
            if name == 'results.txt':
                # print(os.path.join(root, name))
                filename = os.path.join(root, name)
                with open(filename) as f:
                    for line in f:
                        if len(line.strip()) == 0:
                            continue

                        line = line.strip().lower()
                        iteration = filename.split('/')[-2]

                        if '|' in line:
                            model_scores.get(curr).append((iteration, float(line.split('|')[1])))
                        else:
                            curr = 'augm'
                            if line in ['test', 'dev']:
                                curr = line
                            elif line.startswith('next'):
                                curr = 'next_augm'

    print(sorted(model_scores.get('dev'), key=lambda tup: tup[1], reverse=True))
    return sorted(model_scores.get('dev'), key=lambda tup: tup[1], reverse=True)[0][0]


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


