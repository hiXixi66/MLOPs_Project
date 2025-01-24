import torch

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from src.rice_images.model import load_resnet18_timm

model = load_resnet18_timm(num_classes=5)
inputs = torch.randn(5, 3, 224, 224)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./log/resnet18"),
) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

prof.export_chrome_trace("trace2.json")
