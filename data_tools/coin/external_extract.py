"""
Script to extract frames from remote disk
"""

network_path = "hpc:/home/cse/btech/cs1160321/scratch/BTP/COIN/videos"

import subprocess, shlex
import sys, os

def execute(command, stdout=".useless"):
    print (command, '|', stdout)
    command = shlex.split(command)
    with open(stdout, 'w') as outfile:
        process = subprocess.Popen(command, stdout=outfile)

    ec = process.wait()
    if ec != 0:
        exit(ec)

if len(sys.argv) > 1:
    start = int(sys.argv[1])
else:
    start = 0

wd = os.getcwd()
for task_id in range(start, 200):

    all_videos_dir = "/home/arpan/consistency/data/coin/videos"
    # video_dir = "/home/arpan/consistency/data/coin/videos/{}".format(task_id)
    all_frames_dir = "/home/arpan/consistency/data/coin/raw_frames"
    # frames_dir = "/home/arpan/consistency/data/coin/raw_frames/{}".format(task_id)

    os.chdir(all_videos_dir)
    print ("Fetching videos for task_id: {}".format(task_id))
    command = "scp -r {}/{} .".format(network_path, task_id)
    execute(command, "{}.fetch".format(task_id))

    os.chdir(wd)
    print ("Extracting frames")
    command = "bash extract_rgb_frames.sh"
    execute(command, "{}/{}.extract".format(all_videos_dir, task_id))

    os.chdir(all_videos_dir)
    command = "rm -rf {}".format(task_id)
    execute(command, "{}.rmvid".format(task_id))

    os.chdir(all_frames_dir)
    command = "zip -r {0}f.zip {0}".format(task_id)
    execute(command, "{}.zipping".format(task_id))

    print ("Transferring frames")
    command = "scp {}f.zip {}".format(task_id, network_path)
    execute(command, "{}.send".format(task_id, task_id))

    command = "rm -rf {}".format(task_id)
    execute(command, "{}.rmfra".format(task_id))

    command = "rm {}f.zip".format(task_id)
    execute(command, "{}.rmzip".format(task_id))

    os.chdir(wd)
    print ("-------------------------------------------------------------\n")




