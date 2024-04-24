import torch.nn as nn
import torch.multiprocessing as mp
def do_nothing(module):
    return

if __name__ == "__main__":
    mp.set_start_method('spawn')
    module = nn.Linear(4, 4)
    module.cuda()
    module.share_memory() # as far as I can tell this is a no-op?

    print(next(module.parameters()))
    p = mp.Process(target=do_nothing, args=(module,))
    p.start()
    p.join()
    print(next(module.parameters()))