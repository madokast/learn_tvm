"""
python -m example.A03_IRModule_Transformatio
"""
import tvm
from tvm import relax
from example.A02_IRModule import mod_from_torch, params_from_torch, input

mod = mod_from_torch
params = params_from_torch["main"]


def eval(mod, params, input):
    exec = tvm.compile(mod, target="llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, device=dev)

    data = tvm.nd.array(input, device=dev)
    cpu_out = vm["main"](data, *params).numpy()
    """
    [[ 0.11956072 -0.05511119 -0.09557913 -0.04683066 -0.016252   -0.0911963
    0.01005657  0.10374584  0.24086809 -0.08738992]]
    """
    print(cpu_out)


print("====================== A03_IRModule_Transformatio ======================")
mod.show()
print("end mod.show() ----------------")
eval(mod, params, input)

print("====================== Legalizing Relax operations ======================")

mod = relax.transform.LegalizeOps()(mod)
mod.show()
print("end LegalizeOps mod.show() ----------------")
eval(mod, params, input)

print("====================== relax.get_pipeline(\"zero\")(mod) ======================")

mod = relax.get_pipeline("zero")(mod)
mod.show()
eval(mod, params, input)

