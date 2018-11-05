from pickle import load

ARGMAX_PKL_PATH = '/lfs1/joel/experiments/sequence_tagging2/model-3-97-bc2gm/97/10/preds-argmax-97.pkl'
INPUT_FILE_PATH = '/lfs1/joel/experiments/bigmech/data/bc2gm/train-shuf-splits/keep1-rand-occluded/3-97/97-bc2gm-train-occluded.iobes'

OUTPUT_PATH = '/lfs1/joel/experiments/sequence_tagging2/argmaxd_stitched_bc2gm/10/argmaxd.iobes'

# ARGMAX_PKL_PATH = '/lfs1/joel/experiments/sequence_tagging2/temp.pkl'
# INPUT_FILE_PATH = '/lfs1/joel/experiments/sequence_tagging2/temp_in.iob'
# OUTPUT_PATH = '/lfs1/joel/experiments/sequence_tagging2/argmaxd.iob'

TAGS_PATH = '/lfs1/joel/experiments/sequence_tagging2/tags_bc2gm.txt' #'/nas/home/joel/src/sequence_tagging2/data/tags.txt'
token_sep = '\t'

def get_tags_dict(TAGS_PATH):
    key = 0
    tags_dict = {}
    with open(TAGS_PATH) as f:
        for tag in f:
            tags_dict[key] = tag.strip()
            key += 1

    print('Tags found are', tags_dict)
    return tags_dict


'''
Take list of argmax predictions and merge it with the occluded data set
'''
def stitch(INPUT_FILE_PATH, ARGMAX_PKL_PATH, OUTPUT_PATH, tags_dict):
    s_count = 0
    t_count = 0
    res = []
    s = []
    with open(INPUT_FILE_PATH) as in_file,\
         open(ARGMAX_PKL_PATH, 'rb') as argmax_file:
        argmax = load(argmax_file)
        for token in in_file:
            if len(token.strip()) == 0:
                res.append(s[:])
                s_count += 1
                t_count = 0
                s = []
                continue

            token = token.strip().split(token_sep)
            if token[1] == 'O':
                s.append((token[0], tags_dict[int(argmax[s_count][t_count])]))
            else:
                s.append((token[0], token[1]))
            t_count += 1
    res.append(s[:])
    return res


if __name__ == "__main__":
    res = stitch(INPUT_FILE_PATH,
                 ARGMAX_PKL_PATH,
                 OUTPUT_PATH,
                 get_tags_dict(TAGS_PATH))

    print('The length of result array is', len(res))
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n\n'.join(['\n'.join([token_sep.join([word for word in token]) for token in sentence]) for sentence in res]))
