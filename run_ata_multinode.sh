#SBATCH --ntasks=4

#SBATCH --nodes=4

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=96

#SBATCH --partition=train


head_node_ip="192.168.33.77" #fs 4

echo Node IP: $head_node_ip
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/audio_to_audio3b.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 4 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./train.py --job.config_file ${CONFIG_FILE}
dcgmi profile --resume
