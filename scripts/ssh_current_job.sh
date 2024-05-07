#!/bin/bash

# e.g. 6942025.pbs101
oldest_jobid=$(qstat -f -F json | jq '.Jobs | keys[0]' | tr -d '"')

# e.g. x1001c2s4b1n0/1*48
exec_host=$(qstat -f ${oldest_jobid} -F json \
    | jq ".Jobs.\"${oldest_jobid}\"".exec_host \
    | tr -d '"'
)

# e.g. x1001c2s4b1n0
exec_host=${exec_host%%/*}

# e.g. PBS_JOBID=6942025.pbs101 ssh x1001c2s4b1n0
PBS_JOBID=${oldest_jobid} ssh ${exec_host}
