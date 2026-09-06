# exp11: did -ffast-math keep the update-path divisions (acc/denom, override/eps_baked) as fdiv.s?
import glob, json, re, subprocess
OD = "/app/install/llvm/bin/llvm-objdump"; B = "/app/ETH/TrainDeeploy/DeeployTest"
def counts(obj):
    d = subprocess.run(f"{OD} -d {obj}", shell=True, capture_output=True, text=True).stdout
    return len(re.findall(r"\bfdiv\.s", d)), len(re.findall(r"\bfmul\.s", d)), len(re.findall(r"\bfn?m(?:add|sub)\.s", d))
for name in ["deeploymezotest.c.obj", "OptimizerNetwork.c.obj", "TrainingNetwork.c.obj"]:
    objs = glob.glob(f"{B}/**/{name}", recursive=True)
    if not objs: print(f"{name:26s}: not on disk"); continue
    fd, fm, fu = counts(objs[0]); print(f"{name:26s}: fdiv.s={fd:3d} fmul.s={fm:3d} fused={fu:3d}   [{objs[0].split('TEST_SIRACUSA/')[-1][:70]}]")
# fallback/general: does this toolchain+flags turn a runtime-variable division into a reciprocal multiply?
cc = json.load(open(glob.glob(f"{B}/**/compile_commands.json", recursive=True)[0]))
e = next(x for x in cc if x["file"].endswith("/BatchNorm_fp32.c"))
base = re.sub(r"\s-o\s+\S+", "", e["command"]); base = re.sub(r"\s-c\s+\S+$", "", base).strip()
open("/tmp/exp11_div.c", "w").write(
    "typedef float float32_t; float32_t g_over, g_baked;\n"
    "float32_t gproj(float32_t acc, unsigned n){ float32_t denom = 2.0f * 0.01f * (float32_t)n; return acc / denom; }\n"
    "float32_t epsscale(void){ return g_over / g_baked; }\n")
subprocess.run(f"{base} -c /tmp/exp11_div.c -o /tmp/exp11_div.o", shell=True, check=True)
fd, fm, fu = counts("/tmp/exp11_div.o"); print(f"{'snippet (-ffast-math)':26s}: fdiv.s={fd} fmul.s={fm}  (2 source divisions)")
