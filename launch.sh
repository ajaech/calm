#!/usr/bin/bash

python distrib.py \
    --ps_hosts=falcon.ee.washington.edu:2222 \
    --worker_hosts=bird028.ee.washington.edu:2222,bird028.ee.washington.edu:2223 \
    --job_name=ps --task_index=0


python distrib.py \
    --ps_hosts=falcon.ee.washington.edu:2222 \
    --worker_hosts=bird028.ee.washington.edu:2222,bird028.ee.washington.edu:2223 \
    --job_name=worker --task_index=0

python distrib.py \
    --ps_hosts=falcon.ee.washington.edu:2222 \
    --worker_hosts=bird028.ee.washington.edu:2222,bird028.ee.washington.edu:2223 \
    --job_name=worker --task_index=1
