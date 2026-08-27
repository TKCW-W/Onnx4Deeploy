import sys, collections, onnx
import onnx_graphsurgeon as gs
from Deeploy.Targets.PULPOpen.Platform import PULPOptimizer
p = sys.argv[1]
m = onnx.load(p)
# dedup tensor names defensively (graphsurgeon chokes on collisions)
seen = {}
for n in m.graph.node:
    for coll in (n.input, n.output):
        pass
g = gs.import_onnx(m)
g.toposort()
def hist(graph): return dict(collections.Counter(n.op for n in graph.nodes))
print("PRE :", hist(g))
try:
    g2 = PULPOptimizer.optimize(g)
    if isinstance(g2, tuple): g2 = g2[0]
except Exception as e:
    import traceback; traceback.print_exc(); sys.exit(1)
print("POST(raw):", hist(g2))
print("OUTPUTS:", [o.name for o in g2.outputs])
g2.cleanup().toposort()
print("POST:", hist(g2))
h = hist(g2)
print("RequantizedConv=%d float Conv=%d RQSPerturb=%d Quant=%d Dequant=%d RequantShift=%d Gemm=%d BatchNormInternal=%d"
      % (h.get("RequantizedConv",0), h.get("Conv",0), h.get("RQSPerturbRademacher",0),
         h.get("Quant",0), h.get("Dequant",0), h.get("RequantShift",0), h.get("Gemm",0), h.get("BatchNormInternal",0)))
