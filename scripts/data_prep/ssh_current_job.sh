#!/bin/bash

# oldest_jobid=$(qstat -f -F json | jq '.Jobs | keys[0]' | tr -d '"') \
oldest_jobid=7658170.pbs101
exec_host=$(qstat -f ${oldest_jobid} -F json \
    | jq ".Jobs.\"${oldest_jobid}\"".exec_host \
    | tr -d '"'
)
exec_host=${exec_host%%/*}
PBS_JOBID=${oldest_jobid} ssh ${exec_host}

top -u huangyl