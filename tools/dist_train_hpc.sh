#PBS -N SSN
#PBS -l walltime=20:00:00
#Name of job
#Dep name , project name
#PBS -P cse 
#PBS -l select=2:ngpus=2:ncpus=4:centos=skylake
## SPECIFY JOB NOW

## Change to dir from where script was launched

count=0

master=$(head -1 $PBS_NODEFILE)
echo $master

declare -a var
init_count=$count 
while read p; do
      echo $p
      script="(module load compiler/gcc/7.1.0/compilervars && module load compiler/cuda/10.0/compilervars && cd ~/consistency && conda activate consistency4 && python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=${count} --master_addr=\"${master}\" tools/train_localizer.py work_dirs/full_coin/ssn/config.py --launcher pytorch) > /home/cse/btech/cs1160321/consistency/logs/LOG_${count} 2>&1 &"
      echo $script
      ssh -o StrictHostKeyChecking=no -n -f ${USER}@$p $script
      var[`expr $count - $init_count`]=/home/cse/btech/cs1160321/consistency/tools/JACK_$count  
      count=`expr $count + 1`

done <$PBS_NODEFILE

for i in "${var[@]}" 
do
    echo "DON'T KNOW WHAT THIS IS!" 
	echo $i 
    until [ -f $i ]
    do
        sleep 10
    done

done 



