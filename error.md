# 11/20/2024:
Epoch: [1]  [ 180/1217]  eta: 10:57:14  lr: 0.007347  min_lr: 0.007347  loss: 0.9498 (0.9429)  loss_scale: 16384.0000 (16384.0000)  weight_decay: 0.0100 (0.0100)  grad_norm: 0.0062 (0.0071)  time: 35.4895 (0.4691 -- 369.7117)  data: 35.0082 (0.0001 -- 369.2121)  max mem: 12482
Epoch: [1]  [ 200/1217]  eta: 10:43:54  lr: 0.007452  min_lr: 0.007452  loss: 0.9601 (0.9427)  loss_scale: 16384.0000 (16384.0000)  weight_decay: 0.0100 (0.0100)  grad_norm: 0.0070 (0.0071)  time: 37.6397 (0.4750 -- 393.1283)  data: 35.5995 (0.0002 -- 392.6639)  max mem: 12482
Epoch: [1]  [ 220/1217]  eta: 10:27:20  lr: 0.007558  min_lr: 0.007558  loss: 0.9508 (0.9425)  loss_scale: 16384.0000 (16384.0000)  weight_decay: 0.0100 (0.0100)  grad_norm: 0.0099 (0.0084)  time: 35.3953 (0.4723 -- 359.8039)  data: 34.9204 (0.0001 -- 359.3475)  max mem: 12482
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 4045370 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 4045371 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 4045372 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 2) local_rank: 3 (pid: 4045373) of binary: /home/dani/anaconda3/envs/videomaev2/bin/python
Traceback (most recent call last):
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/launch.py", line 193, in <module>
    main()
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/launch.py", line 189, in main
    launch(args)
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/launch.py", line 174, in launch
    run(args)
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/run.py", line 752, in run
    elastic_launch(
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 131, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/dani/anaconda3/envs/videomaev2/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 245, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
run_mae_pretraining.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-11-20_12:41:04
  host      : daniserver
  rank      : 3 (local_rank: 3)
  exitcode  : 2 (pid: 4045373)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================