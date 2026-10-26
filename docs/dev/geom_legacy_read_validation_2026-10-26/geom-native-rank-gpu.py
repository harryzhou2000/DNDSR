import json
import os
import sys

devices=['GPU-c1ca6727-774a-bf06-1299-f60ad9cecec6','GPU-0848655f-4530-ea83-ca9b-c091c5c6f36a']
rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
assert 0 <= rank < len(devices)
os.environ['CUDA_VISIBLE_DEVICES']=devices[rank]
print(json.dumps({'local_rank':rank,'visible_device':devices[rank],'logical_ordinal':0}),flush=True)
os.execv(sys.executable,[sys.executable]+sys.argv[1:])
