#!/usr/bin/env bash

OUTPUT_FILE="/gpfs/fs7/aafc/phenocart/PhenomicsProjects/TrainingoverContainers/cpu_dynamic_monitor.csv"
NUM_SAMPLES=5       # Number of samples per logging interval
SLEEP_BETWEEN=1     # Seconds between samples
TOTAL_CPUS=$(nproc) # Number of logical CPUs (cores)

########################################
# 1) Create CSV header if not exists   #
########################################
if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "Creating CSV file: $OUTPUT_FILE"

    # Build the header row: e.g.
    # Timestamp, CPU0_usr, CPU0_sys, CPU0_iowait, CPU0_idle, CPU0_freq_kHz, CPU0_tempC, CPU1_usr, ...

    echo -n "Timestamp" > "$OUTPUT_FILE"
    for ((cpu=0; cpu<TOTAL_CPUS; cpu++)); do
        echo -n ",CPU${cpu}_usr,CPU${cpu}_sys,CPU${cpu}_iowait,CPU${cpu}_idle,CPU${cpu}_freq_kHz,CPU${cpu}_tempC" >> "$OUTPUT_FILE"
    done
    echo "" >> "$OUTPUT_FILE"
fi

echo "Logging dynamic CPU info to: $OUTPUT_FILE"
echo "Press Ctrl + C to stop."

###################################
# 2) START MONITORING IN A LOOP   #
###################################
while true; do

    # Arrays to sum usage/frequency/temperature for each CPU
    declare -a SUM_USR
    declare -a SUM_SYS
    declare -a SUM_IOW
    declare -a SUM_IDL
    declare -a SUM_FREQ
    declare -a SUM_TEMP

    for ((cpu=0; cpu<TOTAL_CPUS; cpu++)); do
        SUM_USR[$cpu]=0
        SUM_SYS[$cpu]=0
        SUM_IOW[$cpu]=0
        SUM_IDL[$cpu]=0
        SUM_FREQ[$cpu]=0
        SUM_TEMP[$cpu]=0
    done

    # We'll track how many times we've successfully found a sensor match per CPU,
    # in case some cores don't have separate sensor entries
    declare -a TEMP_SAMPLES_COUNT
    for ((cpu=0; cpu<TOTAL_CPUS; cpu++)); do
        TEMP_SAMPLES_COUNT[$cpu]=0
    done

    ####################################################
    # 2a) Collect data NUM_SAMPLES times (averaging)   #
    ####################################################
    for ((sample=1; sample<=NUM_SAMPLES; sample++)); do

        #
        # A) CAPTURE PER-CPU USAGE VIA mpstat
        #
        # mpstat -P ALL 1 1 → 1-second sample for all CPUs
        # We'll parse lines that look like:
        # CPU   %usr  %nice  %sys  %iowait  %irq  %soft  %steal  %guest  %gnice  %idle
        #
        # We'll do a small trick to capture that 1-second output into an array:
        #
        MAPFILE=()
        IFS=$'\n' read -rd '' -a MAPFILE < <(mpstat -P ALL 1 1 2>/dev/null | awk '/^[0-9]/ {print $0}' || true)

        # For each line in MAPFILE, parse usage
        for line in "${MAPFILE[@]}"; do
            # Example line (depending on distro/time format):
            # "Average:     0     3.00   0.00    2.00  0.00   0.00   0.00   0.00   0.00   95.00"
            # or "14:28:00     0     3.00   0.00    2.00  0.00   0.00   0.00   0.00   0.00   95.00"
            tokens=($line)

            # Identify which token is the CPU index
            cpu_id=-1
            for ((t=0; t<${#tokens[@]}; t++)); do
                # If this token is an integer or "all", it's probably the CPU column
                if [[ ${tokens[$t]} =~ ^[0-9]+$ ]] || [[ ${tokens[$t]} == "all" ]]; then
                    cpu_id=${tokens[$t]}
                    break
                fi
            done

            # Skip "all" line if found
            if [[ "$cpu_id" == "all" ]]; then
                continue
            fi

            # If we have a numeric CPU index, parse usage columns
            if [[ "$cpu_id" =~ ^[0-9]+$ ]]; then
                # Typically for mpstat:
                #    CPU   %usr  %nice  %sys  %iowait  %irq  %soft  %steal  %guest  %gnice  %idle
                #
                # We'll assume:
                #   %usr    → tokens[t+1]
                #   %nice   → tokens[t+2]
                #   %sys    → tokens[t+3]
                #   %iowait → tokens[t+4]
                #   %idle   → tokens[t+10]
                # (Might differ slightly by distro; adjust if needed.)

                usr_val="${tokens[$((t+1))]}"
                sys_val="${tokens[$((t+3))]}"
                iow_val="${tokens[$((t+4))]}"
                idl_val="${tokens[$((t+10))]}"

                # Accumulate into arrays
                SUM_USR[$cpu_id]=$(echo "${SUM_USR[$cpu_id]} + $usr_val" | bc)
                SUM_SYS[$cpu_id]=$(echo "${SUM_SYS[$cpu_id]} + $sys_val" | bc)
                SUM_IOW[$cpu_id]=$(echo "${SUM_IOW[$cpu_id]} + $iow_val" | bc)
                SUM_IDL[$cpu_id]=$(echo "${SUM_IDL[$cpu_id]} + $idl_val" | bc)
            fi
        done

        #
        # B) CAPTURE PER-CPU FREQUENCY
        #
        for ((cpu=0; cpu<TOTAL_CPUS; cpu++)); do
            freq_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq"
            if [[ -r "$freq_file" ]]; then
                cur_freq_khz=$(cat "$freq_file" 2>/dev/null)
                # Add to sum
                SUM_FREQ[$cpu]=$(echo "${SUM_FREQ[$cpu]} + ${cur_freq_khz:-0}" | bc)
            fi
        done

        #
        # C) CAPTURE TEMPERATURES (best-effort, mapped by "Core X:")
        #
        # We'll parse each "Core X:" from sensors, e.g.:
        #   Core 0:        +38.0°C  (high = +95.0°C, crit = +105.0°C)
        # The "X" after "Core" typically matches the CPU index.
        #
        # Because we’re sampling multiple times, we can re-parse each second.
        # If your system has a different naming scheme or multiple packages,
        # you’ll need more advanced parsing.
        #
        while IFS= read -r s_line; do
            # Example s_line: "Core 0:        +38.0°C  (high = +95.0°C, crit = +105.0°C)"
            if [[ "$s_line" =~ ^Core[[:space:]]+([0-9]+):.*\+([0-9]+\.[0-9]+) ]]; then
                core_idx="${BASH_REMATCH[1]}"
                core_temp="${BASH_REMATCH[2]}"
                if (( core_idx < TOTAL_CPUS )); then
                    SUM_TEMP[$core_idx]=$(echo "${SUM_TEMP[$core_idx]} + $core_temp" | bc)
                    TEMP_SAMPLES_COUNT[$core_idx]=$(( TEMP_SAMPLES_COUNT[$core_idx] + 1 ))
                fi
            fi
        done < <(sensors 2>/dev/null || true)

        sleep "$SLEEP_BETWEEN"
    done

    ########################################################
    # 2b) Compute AVERAGES for each CPU across the samples #
    ########################################################
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # We'll build one CSV row
    csv_line="$TIMESTAMP"

    for ((cpu=0; cpu<TOTAL_CPUS; cpu++)); do
        # usage is SUM / NUM_SAMPLES
        avg_usr=$(echo "scale=2; ${SUM_USR[$cpu]} / $NUM_SAMPLES" | bc 2>/dev/null)
        avg_sys=$(echo "scale=2; ${SUM_SYS[$cpu]} / $NUM_SAMPLES" | bc 2>/dev/null)
        avg_iow=$(echo "scale=2; ${SUM_IOW[$cpu]} / $NUM_SAMPLES" | bc 2>/dev/null)
        avg_idl=$(echo "scale=2; ${SUM_IDL[$cpu]} / $NUM_SAMPLES" | bc 2>/dev/null)

        # frequency is SUM / NUM_SAMPLES in kHz
        avg_freq=$(echo "scale=0; ${SUM_FREQ[$cpu]} / $NUM_SAMPLES" | bc 2>/dev/null)

        # temperature might not appear in all samples; if we found it N times, average by N
        if (( TEMP_SAMPLES_COUNT[$cpu] > 0 )); then
            avg_temp=$(echo "scale=1; ${SUM_TEMP[$cpu]} / ${TEMP_SAMPLES_COUNT[$cpu]}" | bc 2>/dev/null)
        else
            avg_temp="N/A"
        fi

        # Append these to our CSV line
        csv_line="$csv_line,$avg_usr,$avg_sys,$avg_iow,$avg_idl,$avg_freq,$avg_temp"
    done

    # Write the line to the CSV
    echo "$csv_line" >> "$OUTPUT_FILE"

    # (Optional) print a small summary to terminal
    echo "Logged at: $TIMESTAMP"
done
