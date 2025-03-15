#!/bin/bash

OUTPUT_FILE="/mnt/PhenomicsProjects/TrainingoverContainers/Outputs/system_monitor_log.csv"

# Write CSV header if the file doesn't exist
if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo -n "Timestamp,CPU Load,Used Memory (MB),Available Memory (MB)" > "$OUTPUT_FILE"

    # Check for GPUs and add headers dynamically
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        for ((i=0; i<NUM_GPUS; i++)); do
            echo -n ",GPU${i}_Utilization (%),GPU${i}_Memory Used (MB)" >> "$OUTPUT_FILE"
        done
    fi

    echo "" >> "$OUTPUT_FILE"
fi

NUM_CPUS=$(nproc)
TOTAL_MEM=$(free -m | awk '/Mem:/ {print $2}')
NUM_SAMPLES=10  # Number of samples for averaging

echo "System Monitoring started. Logging to $OUTPUT_FILE..."
echo "Press Ctrl + C to stop."

# Infinite loop to collect data until manually stopped
while true; do
    SUM_USED_MEM=0
    SUM_AVAIL_MEM=0
    SUM_CPU_LOAD=0
    declare -A SUM_GPU_UTIL SUM_GPU_MEM_USED

    # Initialize GPU values
    if command -v nvidia-smi &> /dev/null; then
        for ((i=0; i<NUM_GPUS; i++)); do
            SUM_GPU_UTIL[$i]=0
            SUM_GPU_MEM_USED[$i]=0
        done
    fi

    # Collect Data for 10 seconds
    for ((i=1; i<=NUM_SAMPLES; i++)); do
        CPU_LOAD=$(awk '{print $1}' <<< "$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')")
        if [[ $CPU_LOAD =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            SUM_CPU_LOAD=$(echo "$SUM_CPU_LOAD + $CPU_LOAD" | bc)
        fi

        USED_MEM=$(free -m | awk '/Mem:/ {print $3}')
        AVAIL_MEM=$(free -m | awk '/Mem:/ {print $7}')
        SUM_USED_MEM=$((SUM_USED_MEM + USED_MEM))
        SUM_AVAIL_MEM=$((SUM_AVAIL_MEM + AVAIL_MEM))

        # Capture GPU statistics if available
        if command -v nvidia-smi &> /dev/null; then
            while IFS=, read -r index gpu_util mem_used; do
                index=$(echo "$index" | tr -d ' ')
                gpu_util=$(echo "$gpu_util" | tr -d ' ')
                mem_used=$(echo "$mem_used" | tr -d ' ')

                if [[ $gpu_util =~ ^[0-9]+$ ]]; then
                    SUM_GPU_UTIL[$index]=$((${SUM_GPU_UTIL[$index]} + gpu_util))
                fi
                if [[ $mem_used =~ ^[0-9]+$ ]]; then
                    SUM_GPU_MEM_USED[$index]=$((${SUM_GPU_MEM_USED[$index]} + mem_used))
                fi
            done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
        fi

        sleep 1  # Pause for 1 second before taking the next sample
    done

    # Compute Averages
    if [[ $SUM_CPU_LOAD =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        AVG_CPU_LOAD=$(echo "scale=2; $SUM_CPU_LOAD / $NUM_SAMPLES" | bc)
    else
        AVG_CPU_LOAD="N/A"
    fi

    AVG_USED_MEM=$((SUM_USED_MEM / NUM_SAMPLES))
    AVG_AVAIL_MEM=$((SUM_AVAIL_MEM / NUM_SAMPLES))

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Write data to CSV
    echo -n "$TIMESTAMP,$AVG_CPU_LOAD,$AVG_USED_MEM,$AVG_AVAIL_MEM" >> "$OUTPUT_FILE"

    if command -v nvidia-smi &> /dev/null; then
        for ((i=0; i<NUM_GPUS; i++)); do
            AVG_GPU_UTIL=$((${SUM_GPU_UTIL[$i]} / NUM_SAMPLES))
            AVG_GPU_MEM=$((${SUM_GPU_MEM_USED[$i]} / NUM_SAMPLES))
            echo -n ",$AVG_GPU_UTIL,$AVG_GPU_MEM" >> "$OUTPUT_FILE"
        done
    fi

    echo "" >> "$OUTPUT_FILE"

    echo "Logged at: $TIMESTAMP | CPU Load: $AVG_CPU_LOAD | Mem: ${AVG_USED_MEM}/${TOTAL_MEM} MB"
done
