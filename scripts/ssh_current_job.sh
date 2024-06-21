#!/bin/bash

# e.g. 6942025.pbs101

# oldest_jobid=$(qstat -f -F json | jq '.Jobs | keys[0]' | tr -d '"') \
oldest_jobid=7646422.pbs101
echo $oldest_jobid
# e.g. x1001c2s4b1n0/1*48
exec_host=$(qstat -f ${oldest_jobid} -F json \
    | jq ".Jobs.\"${oldest_jobid}\"".exec_host \
    | tr -d '"'
)
exec_host=${exec_host%%/*}
PBS_JOBID=${oldest_jobid} ssh ${exec_host}

top -u huangyl