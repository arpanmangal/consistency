"""
Script to extract all the zips present in a folder
"""

import argparse
import glob
import shlex
import subprocess

parser = argparse.ArgumentParser(description="Extract zips")
parser.add_argument('in_folder', type=str, help="Path of folder containing all the zips")
parser.add_argument('out_folder', type=str, help="Path of the folder where to extract")
args = parser.parse_args()

for zip_file in glob.glob('%s/*.zip' % args.in_folder):
    # Extract this zip file
    command = "unzip {} -d {}".format(zip_file, args.out_folder)
    print (command)
    command = shlex.split(command)
    process = subprocess.Popen(command)
    