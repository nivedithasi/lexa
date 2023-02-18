import subprocess

devices = range(8)
device_idx = 0

method = "lexa_temporal" # lexa_temporal, lexa_cosine, ddl, diayn, gcsl
task = "robobin" # dmc_walker_walk, dmc_quadruped_run, robobin, kitchen, joint
# camera_distance = 1.86
# azimuth = 90
# elevation = -60
# variation = f"dist{camera_distance}az{azimuth}elev{elevation}_classifier_userobotvideos"
dvd_score_weight = 1.0
variation = f"classifier_userobotvideos_dvdscoreweight{dvd_score_weight}"
# variation = "dist1.86az90elev-60_nodvd_withstate"
# variation = "dist1.5az60elev-5_nodvd_withstate"
# variation = "dist2.0az90elev-5_nodvd_withstate"
# variation = "dist2.0az150elev-5_nodvd_withstate"
# variation = "dist2.5az120elev-5_nodvd_withstate"
# --encoder_cls with_state

logdir = method + '_' + task + '_' + variation
_out = _err = f"/home/ademi_adeniji/lexastuff/lexa_dvd/logs/{logdir}.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
    command = f'CUDA_VISIBLE_DEVICES=2 python /home/ademi_adeniji/lexastuff/lexa_dvd/lexa/train.py --configs defaults lexa_temporal --task {task} ' +\
        f'--logdir /home/ademi_adeniji/lexastuff/lexa_dvd/experiments/{logdir} --dvd_score_weight {dvd_score_weight} --use_robot_videos True'
    print(devices[device_idx%len(devices)], command)
    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
p.wait()
print('returncode', p.returncode)