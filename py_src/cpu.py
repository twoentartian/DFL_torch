import torch

def submit_training_job_cpu(training_node, criterion: torch.nn.CrossEntropyLoss, training_data: torch.Tensor, training_label: torch.Tensor):
    model = training_node.model
    optimizer = training_node.optimizer
    lr_scheduler = training_node.lr_scheduler
    optimizer.zero_grad(set_to_none=True)
    output = model(training_data)
    loss = criterion(output, training_label)
    loss.backward()
    optimizer.step()
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    if lr_scheduler is not None:
        lr_scheduler.step()
    return loss, lrs