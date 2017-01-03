import os

expdir = '/s0/ajaech/exptest'
worker_threads = 8

ps_hosts = (
    'localhost:2233',
    'localhost:2234'
)

worker_hosts = (
    'localhost:2222',
    'localhost:2223',
    'localhost:2224',
    'localhost:2225',
)

print '#!/usr/bin/bash\n'

print 'expdir={0}'.format(expdir)

print 'if [ -d $expdir ] ; then'
print '\trm -rf $expdir'
print 'fi'
print 'mkdir $expdir\n'

boilerplate = (
    'python distrib.py \\',
    '\t--ps_hosts={0} \\'.format(','.join(ps_hosts)),
    '\t--worker_hosts={0} \\'.format(','.join(worker_hosts))
)

print '# Launch the parameter servers\n'

for idx, ps in enumerate(ps_hosts):
    for line in boilerplate:
        print line
    print '\t--job_name=ps --task_index={0} \\'.format(idx)
    print '\t--expdir=$expdir &'
    print '\n\n'

print '# Launch the workers.\n'

for idx, w in enumerate(worker_hosts):
    for line in boilerplate:
        print line
    print '\t--job_name=worker --task_index={0} \\'.format(idx)
    print '\t--expdir=$expdir \\'
    print '\t--worker_threads={0} \\'.format(worker_threads)
    print ' 2> {0} &'.format(os.path.join(expdir, 'errlog.worker{0}'.format(idx)))
    print '\n\n'
