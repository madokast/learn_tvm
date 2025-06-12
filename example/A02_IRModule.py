import numpy as np
import tvm
from tvm import relax

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

# fix torch random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Create a dummy model
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# Give an example argument to torch.export
input = np.random.rand(1, 784).astype("float32")
example_args = (torch.from_numpy(input),)

# Convert the model to IRModule
with torch.no_grad():
    torch_model = TorchModel().eval()
    """
    tensor([[ 0.1196, -0.0551, -0.0956, -0.0468, -0.0163, -0.0912,  0.0101,  0.1037,
          0.2409, -0.0874]])
    """
    output = torch_model(*example_args)
    print(output)
    print('end torch_model(*example_args) ----------------')
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)

# Print the IRModule
"""
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((784, 256), dtype="float32") = R.permute_dims(p_fc1_weight, axes=None)
            lv1: R.Tensor((1, 256), dtype="float32") = R.matmul(x, lv, out_dtype="float32")
            lv2: R.Tensor((1, 256), dtype="float32") = R.add(lv1, p_fc1_bias)
            lv3: R.Tensor((1, 256), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((256, 10), dtype="float32") = R.permute_dims(p_fc2_weight, axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="float32")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, p_fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
"""
mod_from_torch.show()
print("end mod_from_torch.show() ----------------")

"""
{'main': [(256, 784), (256,), (10, 256), (10,)]}
"""
print({k:[(v.shape, v.dtype, v.device) for v in vs] for k, vs in params_from_torch.items()})
print("end params_from_torch ----------------")


mod = mod_from_torch
print(mod.get_global_vars()) # [I.GlobalVar("main")]
print("end mod.get_global_vars() ----------------")

# index by global var name
print(mod["main"])
"""
# from tvm.script import relax as R

@R.function
def main(x: R.Tensor((1, 784), dtype="float32"), p_fc1_weight: R.Tensor((256, 784), dtype="float32"), p_fc1_bias: R.Tensor((256,), dtype="float32"), p_fc2_weight: R.Tensor((10, 256), dtype="float32"), p_fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
    R.func_attr({"num_input": 1})
    with R.dataflow():
        lv: R.Tensor((784, 256), dtype="float32") = R.permute_dims(p_fc1_weight, axes=None)
        lv1: R.Tensor((1, 256), dtype="float32") = R.matmul(x, lv, out_dtype="float32")
        lv2: R.Tensor((1, 256), dtype="float32") = R.add(lv1, p_fc1_bias)
        lv3: R.Tensor((1, 256), dtype="float32") = R.nn.relu(lv2)
        lv4: R.Tensor((256, 10), dtype="float32") = R.permute_dims(p_fc2_weight, axes=None)
        lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="float32")
        lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, p_fc2_bias)
        gv: R.Tensor((1, 10), dtype="float32") = lv6
        R.output(gv)
    return gv
"""
print("end mod['main'] ----------------")

#########################################################


exec = tvm.compile(mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, device=dev)

data = tvm.nd.array(input, device=dev)
cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
"""
[[ 0.11956072 -0.05511119 -0.09557913 -0.04683066 -0.016252   -0.0911963
   0.01005657  0.10374584  0.24086809 -0.08738992]]
"""
print(cpu_out)
