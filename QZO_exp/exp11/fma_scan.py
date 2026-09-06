# SPDX-License-Identifier: MIT
"""exp11 L3 evidence: does the device build FUSE fp32 mul+add (fmadd.s) in the fp32 tail kernels,
and does a scoped strict-fp pragma / -fno-fast-math remove it? Also: which libm symbols the SCE calls.
Run inside the device-build container:  python3 /app/ETH/Onnx4Deeploy/QZO_exp/exp11/fma_scan.py"""
import glob, json, re, subprocess
cc = json.load(open(glob.glob("/app/ETH/TrainDeeploy/DeeployTest/**/compile_commands.json", recursive=True)[0]))
OD = "/app/install/llvm/bin/llvm-objdump"
PRAGMA = "#pragma clang fp contract(off) reassociate(off) reciprocal(off)\n"

def scan(obj, rel=False):
    rflag = "-r" if rel else ""
    d = subprocess.run(f"{OD} -d {rflag} {obj}", shell=True, capture_output=True, text=True).stdout
    fused = len(re.findall(r"\bfn?m(?:add|sub)\.s", d)); sep = len(re.findall(r"\bf(?:mul|add|sub)\.s", d))
    calls = sorted(set(re.findall(r"R_RISCV_CALL(?:_PLT)?\s+([A-Za-z_0-9]+)", d)))
    return fused, sep, calls

def base_cmd(entry):
    b = re.sub(r"\s-o\s+\S+", "", entry["command"]); return re.sub(r"\s-c\s+\S+$", "", b).strip()

for k in ["Gemm", "BatchNorm", "GlobalAveragePool"]:
    e = next(x for x in cc if x["file"].endswith(f"/PULPOpen/src/{k}.c")); base = base_cmd(e); src = e["file"]
    for tag, extra, prepend in [("as-is (-ffast-math)", "", False), ("+ -fno-fast-math", "-fno-fast-math", False), ("+ fp pragma", "", True)]:
        s = src
        if prepend:
            s = f"/tmp/exp11_{k}_pr.c"; open(s, "w").write(PRAGMA + open(src).read())
        r = subprocess.run(f"{base} {extra} -c {s} -o /tmp/exp11_{k}.o", shell=True, capture_output=True, text=True)
        if r.returncode:
            print(f"{k:18s} {tag:22s}: COMPILE FAIL " + r.stderr.strip().splitlines()[-1][:100]); continue
        f, sp, _ = scan(f"/tmp/exp11_{k}.o"); print(f"{k:18s} {tag:22s}: fused={f:3d} separate={sp:3d}")

# SCE-representative snippet: which libm symbols are called (relocations), with and without fast-math
e = next(x for x in cc if x["file"].endswith("/BatchNorm_fp32.c")); base = base_cmd(e)
open("/tmp/exp11_sce.c", "w").write(
    "#include <math.h>\n#include <stdint.h>\ntypedef float float32_t;\n"
    "float32_t sce(const float32_t*lg,uint32_t C,uint32_t lab){float32_t m=lg[0];for(uint32_t j=1;j<C;j++)if(lg[j]>m)m=lg[j];"
    "float32_t s=0.0f;for(uint32_t j=0;j<C;j++)s+=expf(lg[j]-m);float32_t l=logf(s);return -(lg[lab]-m-l);}\n")
for tag, extra in [("-ffast-math", ""), ("-fno-fast-math", "-fno-fast-math")]:
    subprocess.run(f"{base} {extra} -c /tmp/exp11_sce.c -o /tmp/exp11_sce.o", shell=True, check=True)
    f, sp, calls = scan("/tmp/exp11_sce.o", rel=True); print(f"{'SCE snippet':18s} {tag:22s}: fused={f} separate={sp} libm calls={calls}")
