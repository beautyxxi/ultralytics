import torch

from typing import Dict

# 更改总训练轮数为new_epochs
def change_train_epochs_to_resume(ckpt_path : str, new_epochs : int) -> Dict:
    """
    Change the number of epochs to resume training from a checkpoint.
    Args:
        ckpt (Union[str, Dict]): The checkpoint to resume training from.
    Returns:
        Dict: The checkpoint with the number of epochs changed.
    """
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    epoch = ckpt.get("epoch", -1)
    print("last training epoch is %d" % epoch)
    if epoch != -1:
        ckpt["epoch"] = epoch
    else:
        ckpt["epoch"] = ckpt["train_args"]["epochs"] - 1
    ckpt["train_args"]["epochs"] = new_epochs
    torch.save(ckpt, ckpt_path)
    return ckpt