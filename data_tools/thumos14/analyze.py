"""
Script to analyze various aspects of the thumos dataset
"""

from prepare_thumos import load_blocks
import os

def find_class_relations(tag_file, NUM_CLASSES=20):
    print("Relations for:", tag_file)

    relations = dict()
    for c in range(1, NUM_CLASSES + 1):
        relations[str(c)] = set()

    num_single = num_more = num_corrects = 0

    blocks = load_blocks(tag_file)
    for block in blocks:
        seen = set()
        for c in block['correct']:
            c = c[0]
            seen.add(c)
            num_corrects += 1

        for c1 in seen:
            for c2 in seen:
                relations[c1].add(c2)

        if len(seen) == 1:
            num_single += 1
        else:
            num_more += 1

    print (relations, num_single, num_more, num_corrects)

data_root = 'data/thumos14'
train_tag = os.path.join(data_root, 'thumos14_tag_val_normalized_proposal_list.txt')
test_tag = os.path.join(data_root, 'thumos14_tag_test_normalized_proposal_list.txt')

find_class_relations(train_tag)
find_class_relations(test_tag)