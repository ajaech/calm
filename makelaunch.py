
ps_hosts = (
    'bird028.ee.washington.edu:2232', 
#    'bird029.ee.washington.edu:2233'
)

worker_hosts = (
    'bird028.ee.washington.edu:2222', 
    'bird028.ee.washington.edu:2223',
#    'bird029.ee.washington.edu:2224',
#    'bird029.ee.washington.edu:2225'
)

for idx, ps in enumerate(ps_hosts):

    print 'python distrib.py \\'
    print '\t--ps_hosts={0} \\'.format(','.join(ps_hosts))
    print '\t--worker_hosts={0} \\'.format(','.join(worker_hosts))
    print '\t--job_name=ps --task_index={0} \\'.format(idx)
    print '\t--expdir=/g/ssli/transitory/ajaech/exptest'
    print '\n\n'

for idx, w in enumerate(worker_hosts):

    print 'python distrib.py \\'
    print '\t--ps_hosts={0} \\'.format(','.join(ps_hosts))
    print '\t--worker_hosts={0} \\'.format(','.join(worker_hosts))
    print '\t--job_name=worker --task_index={0} \\'.format(idx)
    print '\t--expdir=/g/ssli/transitory/ajaech/exptest'
    print '\n\n'
