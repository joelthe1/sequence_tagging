import os
import time

READIN_DIR = '/lfs1/joel/experiments/sequence_tagging2/model'
OUTPUT_DIR = '/lfs1/joel/experiments/sequence_tagging2/results_{}.txt'\
    .format(str(int(time.time())))

print('Writing results to {} ...'.format(OUTPUT_DIR))

res = {'test': [],
       'dev': [],
       'augm': []}

curr = None

def process_file(filename):
    with open(filename) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            line = line.strip().lower()

            if '|' in line:
                line = ('|' + '|'.join(filename.split('/')[-3:-1])) + line
                if curr == 'augm':
                    line = '|' + val.split(':')[-1].strip() + line
                res.get(curr).append(line)
            else:
                curr = line if line in ['test', 'dev'] else 'augm'
                val = line

                
for root, dirs, files in os.walk(READIN_DIR):
    for name in files:
        if name == 'results.txt':
            # print(os.path.join(root, name))
            process_file(os.path.join(root, name))
            

header = '|Increment|Iteration|F1|Precision|Recall\n'

with open(OUTPUT_DIR, 'w') as f:
    # write test results
    f.write('Test\n')
    f.write(header)
    f.write('\n'.join(sorted(res.get('test'))))

    # write dev results
    f.write('\n\nDev\n')
    f.write(header)
    f.write('\n'.join(sorted(res.get('dev'))))

    # write augment/increment results
    f.write('\n\nIncrement\n')
    f.write('|Tested Increment' + header)
    f.write('\n'.join(sorted(res.get('augm'))) + '\n')


print('Done writing.')
