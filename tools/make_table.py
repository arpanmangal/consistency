"""
Make table from the eval files
"""
import sys
import os
import termtables as tt
import re

eval_path = sys.argv[1]
print (eval_path)
sys.path.append(eval_path)

from eval import evals

# Parse log files and extract task accuracy and map scores
test_scores = []
for name, log_file in evals.items():
    if not os.path.isfile(log_file):
        print ("%s does not exist!" % log_file)
        continue

    for line in open(log_file, 'r'):
        if re.search("Task Classification Accuracy:", line):
            task_acc_line = line
        if re.search("mean AP", line):
            map_line = line

    float_regex = r"[-+]?\d*\.\d+|\d+"
    task_acc = float(re.findall(float_regex, task_acc_line)[0])
    map_scores = [float(s) for s in re.findall(float_regex, map_line)]

    scores = [name, task_acc]+map_scores
    test_scores.append(scores)

string = tt.to_string(
    test_scores,
    header=["# BPs", "Task Classification Acc."]+['mAP @ 0.%d' % i for i in range(1, 10)],
    style=tt.styles.ascii_thin_double,
    padding=(0, 1),
    alignment="c"*11
)
print (string)

