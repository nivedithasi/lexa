import subprocess

devices = range(8)
device_idx = 0

method = "lexa_temporal" # lexa_temporal, lexa_cosine, ddl, diayn, gcsl
task = "kitchen" # dmc_walker_walk, dmc_quadruped_run, robobin, kitchen, joint
variation = "dist1.5az60elev-5_nodvd_v2"
logdir = method + '_' + task + '_' + variation
_out = _err = f"/home/ademi_adeniji/lexastuff/lexa_dvd/logs/{logdir}.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
    command = f'CUDA_VISIBLE_DEVICES=2 python /home/ademi_adeniji/lexastuff/lexa_dvd/lexa/train.py --configs defaults {method} --task {task} --logdir experiments/{logdir}'
    print(devices[device_idx%len(devices)], command)
    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
p.wait()
print('returncode', p.returncode)