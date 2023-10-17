start_parallel() {
    local script_path="$1"
    local n="$2"
    local pid_file="parallel_pids.txt"

    if [[ -z "$script_path" || -z "$n" ]]; then
        echo "Usage: start_parallel <script_path> <number_of_copies>"
        return 1
    fi

    # Clear PID file
    > "$pid_file"

    for i in $(seq 1 "$n"); do
        $script_path > "client_${i}.log" 2>&1 &
        echo $! >> "$pid_file"  # Append the PID to the pid_file
        echo "Started $script_path (copy $i) with PID $!"
    done

    wait
}

kill_parallel() {
    local pid_file="parallel_pids.txt"
    
    if [[ ! -f "$pid_file" ]]; then
        echo "PID file not found. Can't kill processes."
        return 1
    fi

    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then  # Check if process is still running
            echo "Killing process with PID $pid"
            kill "$pid"
        else
            echo "Process with PID $pid not found. It might have already completed."
        fi
    done < "$pid_file"

    # Optional: Clear or remove the PID file after killing processes
    # rm "$pid_file"
}

# Usage:
# start_parallel ./yourScript.sh 16
# ... later when you want to kill them:
# kill_parallel
