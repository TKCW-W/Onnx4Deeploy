# exp11 L3: which mechanism actually yields fused=0 on the real kernels?
import glob, json, re, subprocess
cc = json.load(open(glob.glob("/app/ETH/TrainDeeploy/DeeployTest/**/compile_commands.json", recursive=True)[0]))
OD = "/app/install/llvm/bin/llvm-objdump"
def fused(obj):
    d = subprocess.run(f"{OD} -d {obj}", shell=True, capture_output=True, text=True).stdout
    return len(re.findall(r"\bfn?m(?:add|sub)\.s", d)), len(re.findall(r"\bf(?:mul|add|sub)\.s", d))
def base_cmd(e):
    b = re.sub(r"\s-o\s+\S+", "", e["command"]); return re.sub(r"\s-c\s+\S+$", "", b).strip()
variants = [("-ffp-contract=off", "-ffp-contract=off", None),
            ("-fno-fast-math -ffp-contract=off", "-fno-fast-math -ffp-contract=off", None),
            ("pragma contract(off)", "", "#pragma clang fp contract(off)\n"),
            ("pragma contract+reassoc(off)", "", "#pragma clang fp contract(off) reassociate(off)\n")]
for k in ["Gemm", "BatchNorm"]:
    e = next(x for x in cc if x["file"].endswith(f"/PULPOpen/src/{k}.c")); base = base_cmd(e); src = e["file"]
    for tag, extra, pre in variants:
        s = src
        if pre: s = f"/tmp/exp11_{k}_v.c"; open(s, "w").write(pre + open(src).read())
        r = subprocess.run(f"{base} {extra} -c {s} -o /tmp/exp11_{k}_v.o", shell=True, capture_output=True, text=True)
        if r.returncode:
            err = [l for l in r.stderr.splitlines() if "error" in l][:1]; print(f"{k:10s} {tag:34s}: FAIL {err[0][:90] if err else ''}"); continue
        f, sp = fused(f"/tmp/exp11_{k}_v.o"); print(f"{k:10s} {tag:34s}: fused={f:3d} separate={sp:3d}")
