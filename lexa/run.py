import subprocess

devices = range(8)
device_idx = 0

method = "lexa_temporal" # lexa_temporal, lexa_cosine, ddl, diayn, gcsl
task = "kitchen" # dmc_walker_walk, dmc_quadruped_run, robobin, kitchen, joint
camera_distance = 1.86
azimuth = 90
elevation = -60
variation = f"dist{camera_distance}az{azimuth}elev{elevation}_nodvd_withstate_flag"
# variation = "dist1.86az90elev-60_nodvd_withstate"
# variation = "dist1.5az60elev-5_nodvd_withstate"
# variation = "dist2.0az90elev-5_nodvd_withstate"
# variation = "dist2.0az150elev-5_nodvd_withstate"
# variation = "dist2.5az120elev-5_nodvd_withstate"

logdir = method + '_' + task + '_' + variation
_out = _err = f"/home/ademi_adeniji/lexastuff/lexa_dvd/logs/{logdir}.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
    command = f'CUDA_VISIBLE_DEVICES=0 python /home/ademi_adeniji/lexastuff/lexa_dvd/lexa/train.py --configs defaults {method} --task {task} \
        --logdir /home/ademi_adeniji/lexastuff/lexa_dvd/experiments/{logdir} --encoder_cls with_state --camera_distance {camera_distance} \
        --azimuth {azimuth} --elevation {elevation}'
    print(devices[device_idx%len(devices)], command)
    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
p.wait()
print('returncode', p.returncode)