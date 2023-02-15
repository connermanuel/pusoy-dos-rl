# pusoy-dos-rl
A reinforcement learning model for the Filipino card game Pusoy Dos (a variation of Big Two).

Setup:
* Create an environment with Pytorch and CUDA enabled.
* cd to the root directory and run `pip install -e .`

To train the model, cd into `pusoy` and run `python train.py`. Use the `-h` flag to view CLI args.
To play an interactive game, run `python play.py`.
To observe a model play against dummies, look into `modelvsdummies.py`

I work on Windows, so if something is broken and you aren't on Windows, well, it's probably a Windows thing. 
Though, I did train this on a VM, so this works on debian too, it seems.
