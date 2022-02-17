# Experiment Log

## Exp 1
### resnet20, bs=128, with skips
- run_id: run_20220212-182238

nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 --batch_size 128 > ./training-logs/log-exp1-resnet_20-bs_128-skips_True.txt &

- training stdout: ./training-logs/log-exp1-resnet_20-bs_128-skips_True.txt
#### compute traj

python compute_trajectory.py -s '../results/hw1-exp1/resnet20/run_20220212-182238/ckpts' --direction_file "../results/hw1-exp1/resnet20/run_20220212-182238/fds/buffer.npy.npz" -r "../results/hw1-exp1/resnet20/run_20220212-182238/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20  

- std:
2022-02-13 18:39:54,904 using resnet20 with 269722 parameters
2022-02-13 18:39:54,918 Found 201 models
2022-02-13 18:40:09,278 Dot product is -4.237517714500427e-08
2022-02-13 18:40:10,053 Saving results
2022-02-13 18:40:10,060 xrange: -0.22742463648319244, 0.131802499294281
2022-02-13 18:40:10,065 yrange: -0.06853014975786209, 0.20012108981609344

#### compute surface file
nohup python compute_loss_surface.py     -s "../results/hw1-exp1/resnet20/run_20220212-182238/ckpts/200_model.pt"      --model resnet20      --direction_file "../results/hw1-exp1/resnet20/run_20220212-182238/fds/buffer.npy.npz"     --batch_size 514      --result_folder "../results/hw1-exp1/resnet20/run_20220212-182238/loss_surface"      --surface_file "fd_dir_loss_surface.npz"      --gpu_id 0      --xcoords 51:-0.35:0.35 --ycoords 51:-0.35:0.35   



#### plot traj+loss surface
python plot.py --result_folder "../results/hw1-exp1/resnet20/run_20220212-182238/plots" --trajectory_file "../results/hw1-exp1/resnet20/run_20220212-182238/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj"

- std: figure saved to:  ../results/hw1-exp1/resnet20/run_20220212-182238/plots/resnet20_fd_dir_traj_trajectory_2d

### resnet20, bs=128, no skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 128 \
--remove_skip_connections > ./training-logs/log-exp1-resnet_20-bs_128-skips_False.txt &

- starts: 
- run_id: run_20220212-183317

commands:
#### compute_trajectory
python compute_trajectory.py -s '../results/hw1-exp1/resnet20/run_20220212-183317/ckpts' --direction_file "../results/hw1-exp1/resnet20/run_20220212-183317/fds/buffer.npy.npz" -r "../results/hw1-exp1/resnet20/run_20220212-183317/trajectories" --projection_file fd_dir_proj.npz --model resnet20  --remove_skip_connections

- std:
2022-02-13 18:20:45,354 using resnet20 with 269722 parameters
2022-02-13 18:20:45,367 Found 201 models
2022-02-13 18:20:59,174 Dot product is 3.91155481338501e-08
2022-02-13 18:20:59,636 Saving results
2022-02-13 18:20:59,668 xrange: -0.0868176519870758, 0.3076061010360718
2022-02-13 18:20:59,669 yrange: -0.02207605354487896, 0.3010398745536804
####  plot traj

python plot.py --result_folder "../results/hw1-exp1/resnet20/run_20220212-183317/plots" --trajectory_file "../results/hw1-exp1/resnet20/run_20220212-183317/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj"  

- std: 
figure saved to:  ../results/hw1-exp1/resnet20/run_20220212-183317/plots/resnet20_fd_dir_traj_trajectory_2


### resnet20, bs=64, with skips
- run_id: run_20220212-184503

nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 64  > ./training-logs/log-exp1-resnet_20-bs_64-skips_True.txt &



#### compute traj
python compute_trajectory.py -s '../results/hw1-exp1/resnet20/run_20220212-184503/ckpts' --direction_file "../results/hw1-exp1/resnet20/run_20220212-184503/fds/buffer.npy.npz" -r "../results/hw1-exp1/resnet20/run_20220212-184503/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20 

-std: 
2022-02-13 18:35:08,294 using resnet20 with 269722 parameters
2022-02-13 18:35:08,304 Found 201 models
2022-02-13 18:35:20,926 Dot product is -1.3271346688270569e-08
2022-02-13 18:35:21,360 Saving results
2022-02-13 18:35:21,368 xrange: -0.1321951001882553, 0.25495198369026184
2022-02-13 18:35:21,378 yrange: -0.3281671404838562, 0.3030167520046234


#### compute loss surface
python compute_loss_surface.py     -s "../results/hw1-exp1/resnet20/run_20220212-184503/ckpts/200_model.pt"      --model resnet20      --direction_file "../results/hw1-exp1/resnet20/run_20220212-184503/fds/buffer.npy.npz"     --batch_size 514      --result_folder "../results/hw1-exp1/resnet20/run_20220212-184503/loss_surface"      --surface_file "fd_dir_loss_surface.npz"      --gpu_id 0      --xcoords 51:-0.35:0.35 --ycoords 51:-0.35:0.35  


#### plot traj
python plot.py --result_folder "../results/hw1-exp1/resnet20/run_20220212-184503/plots" --trajectory_file "../results/hw1-exp1/resnet20/run_20220212-184503/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj" 



### resnet20, bs=64, w/o skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 64 --remove_skip_connections > ./training-logs/log-exp1-resnet_20-bs_64-skips_False.txt &

-run_id:  run_20220212-184545
#### compute trajectories
- command:


- std:

#### plot trajectories
- command:


- std:
- std: figure saved to:  ../results/hw1-exp1/resnet20/run_20220212-184503/plots/resnet20_fd_dir_traj_trajectory_2d



---
## Exp1, Run2: Repeat training all four settings second time 
### resnet20, bs=128, with skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1-run2' --mode train  --model resnet20 \
--batch_size 128 > ./training-logs/log-exp1-run2-resnet_20-bs_128_skips_True.txt &

- run_id: run_20220212-184152
#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp1-run2/resnet20/run_20220212-184152/ckpts' --direction_file "../results/hw1-exp1-run2/resnet20/run_20220212-184152/fds/buffer.npy.npz" -r "../results/hw1-exp1-run2/resnet20/run_20220212-184152/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20 

- std:
2022-02-13 20:52:06,186 using resnet20 with 269722 parameters
2022-02-13 20:52:06,187 Found 201 models
2022-02-13 20:52:07,765 Dot product is -6.6356733441352844e-09
2022-02-13 20:52:07,766 The directions are orthogonal
2022-02-13 20:52:07,816 Saving results
2022-02-13 20:52:07,818 xrange: -0.1974458545446396, 0.1605432778596878
2022-02-13 20:52:07,818 yrange: -0.1270122081041336, 0.18138884007930756

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp1-run2/resnet20/run_20220212-184152/plots" --trajectory_file "../results/hw1-exp1-run2/resnet20/run_20220212-184152/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj" 

- std:
figure saved to:  ../results/hw1-exp1-run2/resnet20/run_20220212-184152/plots/resnet20_fd_dir_traj_trajectory_2d

### resnet20, bs=128, no skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1-run2' --mode train  --model resnet20 \
--batch_size 128 \
--remove_skip_connections > ./training-logs/log-exp1-run2-resnet_20-bs_128_skips_False.txt &

- run_id: run_20220212-184314

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp1-run2/resnet20/run_20220212-184314/ckpts' --direction_file "../results/hw1-exp1-run2/resnet20/run_20220212-184314/fds/buffer.npy.npz" -r "../results/hw1-exp1-run2/resnet20/run_20220212-184314/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20  --remove_skip_connections


- std:
2022-02-13 20:57:48,799 using resnet20 with 269722 parameters
2022-02-13 20:57:48,799 Found 201 models
2022-02-13 20:57:50,663 Dot product is 2.1420419216156006e-08
2022-02-13 20:57:50,752 Saving results
2022-02-13 20:57:50,772 xrange: -0.1484358012676239, 0.12508425116539001
2022-02-13 20:57:50,772 yrange: -0.15558530390262604, 0.24489577114582062

#### plot trajectories
- command:

python plot.py --result_folder "../results/hw1-exp1-run2/resnet20/run_20220212-184314/plots" --trajectory_file "../results/hw1-exp1-run2/resnet20/run_20220212-184314/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp1-run2/resnet20/run_20220212-184314/plots/resnet20_fd_dir_traj_trajectory_2d



### resnet20, bs=64, with skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1-run2' --mode train  --model resnet20 \
--batch_size 64  > ./training-logs/log-exp1-run2-resnet_20-bs_64-skips_True.txt &

- run_id: run_20220212-184655
#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp1-run2/resnet20/run_20220212-184655/ckpts' --direction_file "../results/hw1-exp1-run2/resnet20/run_20220212-184655/fds/buffer.npy.npz" -r "../results/hw1-exp1-run2/resnet20/run_20220212-184655/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20 

- std:
2022-02-13 20:55:15,396 using resnet20 with 269722 parameters
2022-02-13 20:55:15,397 Found 201 models
2022-02-13 20:55:17,186 Dot product is -2.7939677238464355e-09
2022-02-13 20:55:17,188 The directions are orthogonal
2022-02-13 20:55:17,223 Saving results
2022-02-13 20:55:17,225 xrange: -0.12142778933048248, 0.35307562351226807
2022-02-13 20:55:17,225 yrange: -0.13371747732162476, 0.3167167603969574

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp1-run2/resnet20/run_20220212-184655/plots" --trajectory_file "../results/hw1-exp1-run2/resnet20/run_20220212-184655/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp1-run2/resnet20/run_20220212-184655/plots/resnet20_fd_dir_traj_trajectory_2d

### resnet20, bs=64, w/o skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1-run2' --mode train  --model resnet20 \
--batch_size 64 --remove_skip_connections > ./training-logs/log-exp1-run2-resnet_20-bs_64-skips_False.txt &

- run_id: run_20220212-184741

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp1-run2/resnet20/run_20220212-184741/ckpts' --direction_file "../results/hw1-exp1-run2/resnet20/run_20220212-184741/fds/buffer.npy.npz" -r "../results/hw1-exp1-run2/resnet20/run_20220212-184741/trajectories" --projection_file "fd_dir_proj.npz" --model resnet20  --remove_skip_connections

- std:
2022-02-13 21:00:17,832 using resnet20 with 269722 parameters
2022-02-13 21:00:17,833 Found 201 models
2022-02-13 21:00:19,883 Dot product is 1.6763806343078613e-08
2022-02-13 21:00:19,985 Saving results
2022-02-13 21:00:19,986 xrange: -0.45412832498550415, 0.2852911949157715
2022-02-13 21:00:19,987 yrange: -0.38012200593948364, 0.2719023823738098


#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp1-run2/resnet20/run_20220212-184741/plots" --trajectory_file "../results/hw1-exp1-run2/resnet20/run_20220212-184741/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet20_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp1-run2/resnet20/run_20220212-184741/plots/resnet20_fd_dir_traj_trajectory_2d

<!---
---
## Exp 2
### resnet20, bs=128, with skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 --batch_size 128 > ./training-logs/log-exp1-resnet_20-bs_128-skips_True.txt &

- run_id: run_20220212-182238
- training stdout: ./training-logs/log-exp1-resnet_20-bs_128-skips_True.txt

### resnet20, bs=128, no skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 128 \
--remove_skip_connections > ./training-logs/log-exp1-resnet_20-bs_128-skips_False.txt &

- starts: 
- run_id: run_20220212-183317
- training stdout: 

### resnet20, bs=64, with skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 64  > ./training-logs/log-exp1-resnet_20-bs_64-skips_True.txt &

- run_id: run_20220212-153021
- training stdout: 


### resnet20, bs=64, w/o skips
nohup python train.py --gpu_id 2 --result_folder '../results/hw1-exp1' --mode train  --model resnet20 \
--batch_size 64 --remove_skip_connections > ./training-logs/log-exp1-resnet_20-bs_64-skips_False.txt &

-run_id:  run_20220212-153157
- training stdout: 
-->



---
## Exp3: Repeat training all four settings with resnet 44
## Exp3-Run1
### resnet44, bs=128, with skips
- run_id: run_20220212-184919

nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run1' --mode train  --model resnet44 \
--batch_size 128 > ./training-logs/log-exp3-run1-resnet_44-bs_128_skips_True.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run1/resnet44/run_20220212-184919/ckpts' --direction_file "../results/hw1-exp3-run1/resnet44/run_20220212-184919/fds/buffer.npy.npz" -r "../results/hw1-exp3-run1/resnet44/run_20220212-184919/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  


- std:
2022-02-13 21:24:31,020 using resnet44 with 658586 parameters
2022-02-13 21:24:31,021 Found 201 models
2022-02-13 21:24:36,357 Dot product is 1.862645149230957e-08
2022-02-13 21:24:36,593 Saving results
2022-02-13 21:24:36,594 xrange: -0.0853467583656311, 0.1545078456401825
2022-02-13 21:24:36,594 yrange: -0.08918006718158722, 0.2658706605434418

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run1/resnet44/run_20220212-184919/plots" --trajectory_file "../results/hw1-exp3-run1/resnet44/run_20220212-184919/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp3-run1/resnet44/run_20220212-184919/plots/resnet44_fd_dir_traj_trajectory_2d


### resnet44, bs=128, no skips
- run_id: run_20220212-185151
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run1' --mode train  --model resnet44 \
--batch_size 128 \
--remove_skip_connections > ./training-logs/log-exp3-run1-resnet_44-bs_128_skips_False.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run1/resnet44/run_20220212-185151/ckpts' --direction_file "../results/hw1-exp3-run1/resnet44/run_20220212-185151/fds/buffer.npy.npz" -r "../results/hw1-exp3-run1/resnet44/run_20220212-185151/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  --remove_skip_connections


- std:
2022-02-13 21:28:46,543 using resnet44 with 658586 parameters
2022-02-13 21:28:46,544 Found 201 models
2022-02-13 21:28:52,975 Dot product is -5.238689482212067e-10
2022-02-13 21:28:52,975 The directions are orthogonal
2022-02-13 21:28:53,046 Saving results
2022-02-13 21:28:53,047 xrange: -0.402132123708725, 0.15674664080142975
2022-02-13 21:28:53,047 yrange: -0.3902987837791443, 0.4776747226715088
#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run1/resnet44/run_20220212-185151/plots" --trajectory_file "../results/hw1-exp3-run1/resnet44/run_20220212-185151/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp3-run1/resnet44/run_20220212-185151/plots/resnet44_fd_dir_traj_trajectory_2d


### resnet44, bs=64, with skips
- run_id: run_20220212-185216
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run1' --mode train  --model resnet44 \
--batch_size 64  > ./training-logs/log-exp3-run1-resnet_44-bs_64-skips_True.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run1/resnet44/run_20220212-185216/ckpts' --direction_file "../results/hw1-exp3-run1/resnet44/run_20220212-185216/fds/buffer.npy.npz" -r "../results/hw1-exp3-run1/resnet44/run_20220212-185216/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  


- std:

2022-02-13 21:26:45,635 using resnet44 with 658586 parameters
2022-02-13 21:26:45,636 Found 201 models
2022-02-13 21:26:51,456 Dot product is 1.5366822481155396e-08
2022-02-13 21:26:51,736 Saving results
2022-02-13 21:26:51,737 xrange: -0.1904318928718567, 0.190105140209198
2022-02-13 21:26:51,737 yrange: -0.2791413366794586, 0.4509590268135071

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run1/resnet44/run_20220212-185216/plots" --trajectory_file "../results/hw1-exp3-run1/resnet44/run_20220212-185216/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp3-run1/resnet44/run_20220212-185216/plots/resnet44_fd_dir_traj_trajectory_2d


### resnet44, bs=64, w/o skips
- run_id: run_20220212-185256
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run1' --mode train  --model resnet44 \
--batch_size 64 --remove_skip_connections > ./training-logs/log-exp3-run1-resnet_44-bs_64-skips_False.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run1/resnet44/run_20220212-185256/ckpts' --direction_file "../results/hw1-exp3-run1/resnet44/run_20220212-185256/fds/buffer.npy.npz" -r "../results/hw1-exp3-run1/resnet44/run_20220212-185256/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  --remove_skip_connections


- std:
2022-02-13 21:30:40,644 using resnet44 with 658586 parameters
2022-02-13 21:30:40,645 Found 201 models
2022-02-13 21:30:45,765 Dot product is -5.587935447692871e-09
2022-02-13 21:30:45,766 The directions are orthogonal
2022-02-13 21:30:45,835 Saving results
2022-02-13 21:30:45,836 xrange: -0.17813749611377716, 0.9555846452713013
2022-02-13 21:30:45,836 yrange: -1.301344633102417, 0.3389054834842682


#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run1/resnet44/run_20220212-185256/plots" --trajectory_file "../results/hw1-exp3-run1/resnet44/run_20220212-185256/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp3-run1/resnet44/run_20220212-185256/plots/resnet44_fd_dir_traj_trajectory_2d

## Exp3-Run2

### resnet44, bs=128, with skips
- run_id: run_20220212-185603
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run2' --mode train  --model resnet44 \
--batch_size 128 > ./training-logs/log-exp3-run2-resnet_44-bs_128_skips_True.txt &


#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run2/resnet44/run_20220212-185603/ckpts' --direction_file "../results/hw1-exp3-run2/resnet44/run_20220212-185603/fds/buffer.npy.npz" -r "../results/hw1-exp3-run2/resnet44/run_20220212-185603/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  

- std:
2022-02-13 21:38:03,961 using resnet44 with 658586 parameters
2022-02-13 21:38:03,962 Found 201 models
2022-02-13 21:38:10,219 Dot product is 4.190951585769653e-09
2022-02-13 21:38:10,220 The directions are orthogonal
2022-02-13 21:38:10,288 Saving results
2022-02-13 21:38:10,289 xrange: -0.13527707755565643, 0.20756691694259644
2022-02-13 21:38:10,289 yrange: -0.10163739323616028, 0.22461237013339996

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run2/resnet44/run_20220212-185603/plots" --trajectory_file "../results/hw1-exp3-run2/resnet44/run_20220212-185603/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp3-run2/resnet44/run_20220212-185603/plots/resnet44_fd_dir_traj_trajectory_2d


### resnet44, bs=128, no skips
- run_id: run_20220212-185701
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run2' --mode train  --model resnet44 \
--batch_size 128 \
--remove_skip_connections > ./training-logs/log-exp3-run2-resnet_44-bs_128_skips_False.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run2/resnet44/run_20220212-185701/ckpts' --direction_file "../results/hw1-exp3-run2/resnet44/run_20220212-185701/fds/buffer.npy.npz" -r "../results/hw1-exp3-run2/resnet44/run_20220212-185701/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  --remove_skip_connections

- std:
2022-02-13 21:42:21,608 using resnet44 with 658586 parameters
2022-02-13 21:42:21,609 Found 201 models
2022-02-13 21:42:28,264 Dot product is 1.2281816452741623e-08
2022-02-13 21:42:28,484 Saving results
2022-02-13 21:42:28,485 xrange: -0.03859204426407814, 0.3297621011734009
2022-02-13 21:42:28,485 yrange: -0.28024208545684814, 0.09017503261566162

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run2/resnet44/run_20220212-185701/plots" --trajectory_file "../results/hw1-exp3-run2/resnet44/run_20220212-185701/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp3-run2/resnet44/run_20220212-185701/plots/resnet44_fd_dir_traj_trajectory_2d


### resnet44, bs=64, with skips
- run_id: run_20220212-185915
nohup python train.py --gpu_id 3 --result_folder '../results/hw1-exp3-run2' --mode train  --model resnet44 \
--batch_size 64  > ./training-logs/log-exp3-run2-resnet_44-bs_64-skips_True.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run2/resnet44/run_20220212-185915/ckpts' --direction_file "../results/hw1-exp3-run2/resnet44/run_20220212-185915/fds/buffer.npy.npz" -r "../results/hw1-exp3-run2/resnet44/run_20220212-185915/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  

- std:
2022-02-13 21:40:28,863 using resnet44 with 658586 parameters
2022-02-13 21:40:28,863 Found 201 models
2022-02-13 21:40:34,188 Dot product is -8.149072527885437e-09
2022-02-13 21:40:34,189 The directions are orthogonal
2022-02-13 21:40:34,244 Saving results
2022-02-13 21:40:34,245 xrange: -0.5222665667533875, 0.2741740643978119
2022-02-13 21:40:34,245 yrange: -0.24932898581027985, 0.23600134253501892

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run2/resnet44/run_20220212-185915/plots" --trajectory_file "../results/hw1-exp3-run2/resnet44/run_20220212-185915/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp3-run2/resnet44/run_20220212-185915/plots/resnet44_fd_dir_traj_trajectory_2d

### resnet44, bs=64, w/o skips
- run_id: run_20220212-190006
nohup python train.py --gpu_id 0 --result_folder '../results/hw1-exp3-run2' --mode train  --model resnet44 \
--batch_size 64 --remove_skip_connections > ./training-logs/log-exp3-run2-resnet_44-bs_64-skips_False.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp3-run2/resnet44/run_20220212-190006/ckpts' --direction_file "../results/hw1-exp3-run2/resnet44/run_20220212-190006/fds/buffer.npy.npz" -r "../results/hw1-exp3-run2/resnet44/run_20220212-190006/trajectories" --projection_file "fd_dir_proj.npz" --model resnet44  --remove_skip_connections

- std:
2022-02-13 21:44:23,501 using resnet44 with 658586 parameters
2022-02-13 21:44:23,502 Found 201 models
2022-02-13 21:44:28,378 Dot product is -9.313225746154785e-09
2022-02-13 21:44:28,378 The directions are orthogonal
2022-02-13 21:44:28,442 Saving results
2022-02-13 21:44:28,443 xrange: -0.6423287987709045, 0.06748200953006744
2022-02-13 21:44:28,443 yrange: -0.34845295548439026, 0.4299986958503723

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp3-run2/resnet44/run_20220212-190006/plots" --trajectory_file "../results/hw1-exp3-run2/resnet44/run_20220212-190006/trajectories/fd_dir_proj.npz"  --plot_prefix "resnet44_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp3-run2/resnet44/run_20220212-190006/plots/resnet44_fd_dir_traj_trajectory_2d

---
## Exp 4: fully-connected layers(fcnet3)

### fcnet3, n_hidden=64,  act=relu, lr=0.01, bs=128
- run_id: run_20220213-134945
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp4-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 128 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_128.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp4-run1/fcnet3/run_20220213-134945/ckpts' --direction_file "../results/hw1-exp4-run1/fcnet3/run_20220213-134945/fds/buffer.npy.npz" -r "../results/hw1-exp4-run1/fcnet3/run_20220213-134945/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 22:17:38,851 using fcnet3 with 201482 parameters
2022-02-13 22:17:38,851 Found 201 models
2022-02-13 22:17:39,139 Dot product is -6.705522537231445e-08
2022-02-13 22:17:39,210 Saving results
2022-02-13 22:17:39,211 xrange: -0.0523371696472168, 0.061990853399038315
2022-02-13 22:17:39,212 yrange: -0.08054402470588684, 0.09352876245975494

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp4-run1/fcnet3/run_20220213-134945/plots" --trajectory_file "../results/hw1-exp4-run1/fcnet3/run_20220213-134945/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp4-run1/fcnet3/run_20220213-134945/plots/fcnet3_fd_dir_traj_trajectory_2d

### fcnet3, n_hidden=64,  act=relu, lr=0.01, bs=64
- run_id: run_20220213-140335
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp4-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 64 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_64.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp4-run1/fcnet3/run_20220213-140335/ckpts' --direction_file "../results/hw1-exp4-run1/fcnet3/run_20220213-140335/fds/buffer.npy.npz" -r "../results/hw1-exp4-run1/fcnet3/run_20220213-140335/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 22:23:05,083 using fcnet3 with 201482 parameters
2022-02-13 22:23:05,084 Found 201 models
2022-02-13 22:23:05,315 Dot product is -2.2351741790771484e-08
2022-02-13 22:23:05,382 Saving results
2022-02-13 22:23:05,383 xrange: -0.055752016603946686, 0.11581733077764511
2022-02-13 22:23:05,383 yrange: -0.0836544930934906, 0.1347123682498932

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp4-run1/fcnet3/run_20220213-140335/plots" --trajectory_file "../results/hw1-exp4-run1/fcnet3/run_20220213-140335/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp4-run1/fcnet3/run_20220213-140335/plots/fcnet3_fd_dir_traj_trajectory_2d



### fcnet3, n_hidden=64,  act=relu, lr=0.01, bs=256
- run_id: run_20220213-140444
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp4-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 256 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_256.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp4-run1/fcnet3/run_20220213-140444/ckpts' --direction_file "../results/hw1-exp4-run1/fcnet3/run_20220213-140444/fds/buffer.npy.npz" -r "../results/hw1-exp4-run1/fcnet3/run_20220213-140444/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 22:25:59,536 using fcnet3 with 201482 parameters
2022-02-13 22:25:59,537 Found 201 models
2022-02-13 22:25:59,757 Dot product is -8.381903171539307e-09
2022-02-13 22:25:59,757 The directions are orthogonal
2022-02-13 22:25:59,766 Saving results
2022-02-13 22:25:59,767 xrange: -0.027735847979784012, 0.10357309132814407
2022-02-13 22:25:59,767 yrange: -0.04855641350150108, 0.03406490385532379

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp4-run1/fcnet3/run_20220213-140444/plots" --trajectory_file "../results/hw1-exp4-run1/fcnet3/run_20220213-140444/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp4-run1/fcnet3/run_20220213-140444/plots/fcnet3_fd_dir_traj_trajectory_2d

---
## Exp 5: fully-connected layers(fcnet3)

### fcnet3, n_hidden=64,  act=relu, lr=0.001, bs=64
- run_id: run_20220213-141524
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp5-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 64 --lr 1e-3 > ./training-logs/log-exp5-run1-fcnet_3-n_hidden_64-act_relu_bs_64-lr_1e-3.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp5-run1/fcnet3/run_20220213-141524/ckpts' --direction_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141524/fds/buffer.npy.npz" -r "../results/hw1-exp5-run1/fcnet3/run_20220213-141524/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 22:58:20,715 using fcnet3 with 201482 parameters
2022-02-13 22:58:20,716 Found 201 models
2022-02-13 22:58:20,935 Dot product is 3.725290298461914e-08
2022-02-13 22:58:20,997 Saving results
2022-02-13 22:58:20,997 xrange: -0.06359977275133133, 0.025867994874715805
2022-02-13 22:58:20,998 yrange: -0.015723884105682373, 0.11572041362524033

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp5-run1/fcnet3/run_20220213-141524/plots" --trajectory_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141524/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp5-run1/fcnet3/run_20220213-141524/plots/fcnet3_fd_dir_traj_trajectory_2d

### fcnet3, n_hidden=64,  act=relu, lr=0.001, bs=128
- run_id: run_20220213-141625
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp5-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 128  --lr 1e-3 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_128-lr_1e-3.txt &


#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp5-run1/fcnet3/run_20220213-141625/ckpts' --direction_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141625/fds/buffer.npy.npz" -r "../results/hw1-exp5-run1/fcnet3/run_20220213-141625/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 23:02:49,589 using fcnet3 with 201482 parameters
2022-02-13 23:02:49,590 Found 201 models
2022-02-13 23:02:49,841 Dot product is 3.5390257835388184e-08
2022-02-13 23:02:49,913 Saving results
2022-02-13 23:02:49,914 xrange: -0.11243981122970581, 0.013777435757219791
2022-02-13 23:02:49,914 yrange: -0.01174354087561369, 0.06108918786048889

#### plot trajectories
- command:python plot.py --result_folder "../results/hw1-exp5-run1/fcnet3/run_20220213-141625/plots" --trajectory_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141625/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp5-run1/fcnet3/run_20220213-141625/plots/fcnet3_fd_dir_traj_trajectory_2d

### fcnet3, n_hidden=64,  act=relu, lr=0.001, bs=256
- run_id: run_20220213-141652
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp5-run1' --mode train  \
--model fcnet3 --n_hidden 64 --act relu \
--batch_size 256  --lr 1e-3 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_256-lr_1e-3.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp5-run1/fcnet3/run_20220213-141652/ckpts' --direction_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141652/fds/buffer.npy.npz" -r "../results/hw1-exp5-run1/fcnet3/run_20220213-141652/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 64 --act relu  

- std:
2022-02-13 23:04:24,905 using fcnet3 with 201482 parameters
2022-02-13 23:04:24,906 Found 201 models
2022-02-13 23:04:25,139 Dot product is 6.868503987789154e-09
2022-02-13 23:04:25,140 The directions are orthogonal
2022-02-13 23:04:25,149 Saving results
2022-02-13 23:04:25,150 xrange: -0.06272587180137634, 0.012043699622154236
2022-02-13 23:04:25,150 yrange: -0.005826443899422884, 0.03449536859989166

#### plot trajectories
- command:python plot.py --result_folder "../results/hw1-exp5-run1/fcnet3/run_20220213-141652/plots" --trajectory_file "../results/hw1-exp5-run1/fcnet3/run_20220213-141652/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp5-run1/fcnet3/run_20220213-141652/plots/fcnet3_fd_dir_traj_trajectory_2d

---
## Exp 6: fcnet3

### fcnet3, n_hidden=128,  act=relu, lr=0.001, bs=64
- run_id: run_20220213-142117

nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp6-run1' --mode train  \
--model fcnet3 --n_hidden 128 --act relu \
--batch_size 64 --lr 1e-3 > ./training-logs/log-exp6-run1-fcnet_3-n_hidden_128-act_relu_bs_64-lr_1e-3.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp6-run1/fcnet3/run_20220213-142117/ckpts' --direction_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142117/fds/buffer.npy.npz" -r "../results/hw1-exp6-run1/fcnet3/run_20220213-142117/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 128 --act relu  

- std:
2022-02-13 23:17:38,967 using fcnet3 with 411146 parameters
2022-02-13 23:17:38,968 Found 201 models
2022-02-13 23:17:39,287 Dot product is 6.984919309616089e-09
2022-02-13 23:17:39,287 The directions are orthogonal
2022-02-13 23:17:39,304 Saving results
2022-02-13 23:17:39,306 xrange: -0.006118496414273977, 0.16924762725830078
2022-02-13 23:17:39,306 yrange: -0.019344870001077652, 0.046168502420186996

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp6-run1/fcnet3/run_20220213-142117/plots" --trajectory_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142117/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
../results/hw1-exp6-run1/fcnet3/run_20220213-142117/plots/fcnet3_fd_dir_traj_trajectory_2d

### fcnet3, n_hidden=128,  act=relu, lr=0.001, bs=128
- run_id: run_20220213-142214
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp6-run1' --mode train  \
--model fcnet3 --n_hidden 128 --act relu \
--batch_size 128  --lr 1e-3 > ./training-logs/log-exp6-run1-fcnet_3-n_hidden_128-act_relu_bs_128-lr_1e-3.txt &



#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp6-run1/fcnet3/run_20220213-142214/ckpts' --direction_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142214/fds/buffer.npy.npz" -r "../results/hw1-exp6-run1/fcnet3/run_20220213-142214/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 128 --act relu  


- std:
2022-02-13 23:20:17,878 using fcnet3 with 411146 parameters
2022-02-13 23:20:17,879 Found 201 models
2022-02-13 23:20:18,246 Dot product is -1.30385160446167e-08
2022-02-13 23:20:18,370 Saving results
2022-02-13 23:20:18,371 xrange: -0.008466974832117558, 0.08531135320663452
2022-02-13 23:20:18,371 yrange: -0.007423770613968372, 0.06332957744598389

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp6-run1/fcnet3/run_20220213-142214/plots" --trajectory_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142214/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp6-run1/fcnet3/run_20220213-142214/plots/fcnet3_fd_dir_traj_trajectory_2d



### fcnet3, n_hidden=128,  act=relu, lr=0.001, bs=256
- run_id: run_20220213-142255
nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp6-run1' --mode train  \
--model fcnet3 --n_hidden 128 --act relu \
--batch_size 256  --lr 1e-3 > ./training-logs/log-exp6-run1-fcnet_3-n_hidden_128-act_relu_bs_256-lr_1e-3.txt &


#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp6-run1/fcnet3/run_20220213-142255/ckpts' --direction_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142255/fds/buffer.npy.npz" -r "../results/hw1-exp6-run1/fcnet3/run_20220213-142255/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet3  --n_hidden 128 --act relu  

- std:
2022-02-13 23:21:53,519 using fcnet3 with 411146 parameters
2022-02-13 23:21:53,520 Found 201 models
2022-02-13 23:21:53,837 Dot product is -4.470348358154297e-08
2022-02-13 23:21:53,971 Saving results
2022-02-13 23:21:53,973 xrange: -0.011356432922184467, 0.060731444507837296
2022-02-13 23:21:53,973 yrange: -0.01941930130124092, 0.011100438423454762

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp6-run1/fcnet3/run_20220213-142255/plots" --trajectory_file "../results/hw1-exp6-run1/fcnet3/run_20220213-142255/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet3_fd_dir_traj"   


- std:
figure saved to:  ../results/hw1-exp6-run1/fcnet3/run_20220213-142255/plots/fcnet3_fd_dir_traj_trajectory_2d




---
## Exp 7: fcnet5

### fcnet5, n_hidden=64,  act=relu, lr=0.001, bs=64
- run_id: run_20220213-143528
nohup python train-fc.py --gpu_id 3 --result_folder '../results/hw1-exp7-run1' --mode train  \
--model fcnet5 --n_hidden 64 --act relu \
--batch_size 64 --lr 1e-3 > ./training-logs/log-exp7-run1-fcnet_5-n_hidden_64-act_relu_bs_64-lr_1e-3.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp7-run1/fcnet5/run_20220213-143528/ckpts' --direction_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143528/fds/buffer.npy.npz" -r "../results/hw1-exp7-run1/fcnet5/run_20220213-143528/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet5  --n_hidden 64 --act relu  

- std:
2022-02-13 23:29:18,143 using fcnet5 with 209802 parameters
2022-02-13 23:29:18,143 Found 201 models
2022-02-13 23:29:18,411 Dot product is -1.862645149230957e-09
2022-02-13 23:29:18,411 The directions are orthogonal
2022-02-13 23:29:18,421 Saving results
2022-02-13 23:29:18,422 xrange: -0.10831791162490845, 0.016515132039785385
2022-02-13 23:29:18,422 yrange: -0.012224673293530941, 0.05959323048591614

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp7-run1/fcnet5/run_20220213-143528/plots" --trajectory_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143528/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet5_fd_dir_traj"   

- std:
../results/hw1-exp7-run1/fcnet5/run_20220213-143528/plots/fcnet5_fd_dir_traj_trajectory_2d


### fcnet5, n_hidden=64,  act=relu, lr=0.001, bs=128
- run_id: run_20220213-143700
nohup python train-fc.py --gpu_id 3 --result_folder '../results/hw1-exp7-run1' --mode train  \
--model fcnet5 --n_hidden 64 --act relu \
--batch_size 128  --lr 1e-3 > ./training-logs/log-exp7-run1-fcnet_5-n_hidden_64-act_relu_bs_128-lr_1e-3.txt &


#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp7-run1/fcnet5/run_20220213-143700/ckpts' --direction_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143700/fds/buffer.npy.npz" -r "../results/hw1-exp7-run1/fcnet5/run_20220213-143700/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet5  --n_hidden 64 --act relu  

- std:
2022-02-13 23:31:18,969 using fcnet5 with 209802 parameters
2022-02-13 23:31:18,970 Found 201 models
2022-02-13 23:31:19,235 Dot product is -4.6333298087120056e-08
2022-02-13 23:31:19,300 Saving results
2022-02-13 23:31:19,301 xrange: -0.004572638310492039, 0.0788453221321106
2022-02-13 23:31:19,302 yrange: -0.0595209077000618, 0.02683810144662857

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp7-run1/fcnet5/run_20220213-143700/plots" --trajectory_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143700/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet5_fd_dir_traj"   


- std:
../results/hw1-exp7-run1/fcnet5/run_20220213-143700/plots/fcnet5_fd_dir_traj_trajectory_2d

### fcnet5, n_hidden=64,  act=relu, lr=0.001, bs=256
- run_id: run_20220213-143734
nohup python train-fc.py --gpu_id 3 --result_folder '../results/hw1-exp7-run1' --mode train  \
--model fcnet5 --n_hidden 64 --act relu \
--batch_size 256  --lr 1e-3 > ./training-logs/log-exp7-run1-fcnet_5-n_hidden_64-act_relu_bs_256-lr_1e-3.txt &

#### compute trajectories
- command:
python compute_trajectory.py -s '../results/hw1-exp7-run1/fcnet5/run_20220213-143734/ckpts' --direction_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143734/fds/buffer.npy.npz" -r "../results/hw1-exp7-run1/fcnet5/run_20220213-143734/trajectories" --projection_file "fd_dir_proj.npz" --model fcnet5  --n_hidden 64 --act relu  

- std:
2022-02-13 23:32:56,063 using fcnet5 with 209802 parameters
2022-02-13 23:32:56,064 Found 201 models
2022-02-13 23:32:56,330 Dot product is -1.210719347000122e-08
2022-02-13 23:32:56,399 Saving results
2022-02-13 23:32:56,400 xrange: -0.0026694838888943195, 0.05420742183923721
2022-02-13 23:32:56,400 yrange: -0.0062677087262272835, 0.07753381133079529

#### plot trajectories
- command:
python plot.py --result_folder "../results/hw1-exp7-run1/fcnet5/run_20220213-143734/plots" --trajectory_file "../results/hw1-exp7-run1/fcnet5/run_20220213-143734/trajectories/fd_dir_proj.npz"  --plot_prefix "fcnet5_fd_dir_traj"   

- std:
figure saved to:  ../results/hw1-exp7-run1/fcnet5/run_20220213-143734/plots/fcnet5_fd_dir_traj_trajectory_2d

---
## analysis commands
- for pca direction (the statefile folder is folder of all checkpoints)
python create_directions.py \
--model resnet20
--statefile_folder '../results/hw1-exp3-run1/resnet20/run_20220212-184545/ckpts/'  \
-r results/resnet20_skip_bn_bias \
    --direction_file pca_directions.npz  \
    
    
### Computing Optimization Trajectories:
- Input: 
a folder of checkpoints (-s argument), 
precomputed gradient's directions (npz file)

- Output: 
The results are saved to the projection file under the to-be-created "-r (ie. trajectory) folder"
- Command: 

```python
python compute_trajectory.py \
-s '../results/hw1-exp1/resnet20/run_20220212-184545/ckpts/' \
--direction_file "../results/hw1-exp1/resnet20/run_20220212-184545/fds/buffer.npy.npz" \
-r "../results/hw1-exp1/resnet20/run_20220212-184545/trajectories" \
--projection_file fd_dir_proj.npz --model resnet20 
```

```shell
#output
2022-02-13 14:57:01,098 using resnet20 with 269722 parameters
2022-02-13 14:57:01,104 Found 201 models
2022-02-13 14:57:18,201 Dot product is 1.6763806343078613e-08
2022-02-13 14:57:18,882 Saving results
2022-02-13 14:57:18,925 xrange: -0.18108321726322174, 0.3358021676540375
2022-02-13 14:57:18,930 yrange: -0.11699468642473221, 0.3524200916290283
```

### Computing loss landscapes of final models
Creates a new folder `loss_surface` inside the run-<run_id> folder and save the file as 
argument of `--surface_file` 

# very slow!!!
## status: 
- 3:15pm: started this 
- 5:48pm: still running 

## pid: 20710
```python
nohup \
python compute_loss_surface.py \
--result_folder results/resnet20_skip_bn_bias_remove_skip_connections/loss_surface/  \
-s "../results/hw1-exp1/resnet20/run_20220212-184545/ckpts/200_model.pt"  \
--batch_size 1000  \
--model resnet20  \
--direction_file "../results/hw1-exp1/resnet20/run_20220212-184545/fds/buffer.npy.npz" \
--surface_file fd_dir_loss_surface.npz \
--gpu_id 0 \
--xcoords 51:-10:40 --ycoords 51:-10:40  > log-compute-loss-surface.txt&
                    
```

### Plotting results:
You can pass either trajectory file or surface file or both in the command below.

```python

python plot.py --result_folder figures/resnet56/ \
--trajectory_file r"../results/hw1-exp1/resnet20/run_20220212-184545/trajectories/fd_dir_proj.npz"  \
--surface_file "../results/hw1-exp1/resnet20/run_20220212-184545/loss_surface/fd_dir_loss_surface.npz" \
--plot_prefix resnet20_fd_dir

```

#### just trajectory file
```python
python plot.py --result_folder "../results/hw1-exp1/resnet20/run_20220212-184545/plots" \
--trajectory_file "../results/hw1-exp1/resnet20/run_20220212-184545/trajectories/fd_dir_proj.npz"  \
--plot_prefix resnet20_fd_dir_traj
```