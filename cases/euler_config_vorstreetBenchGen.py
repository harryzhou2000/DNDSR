import os
import json
from collections import OrderedDict
import copy

forceWrite = True
ignoreExist = True
baseConfName = "./euler_config_vorstreetBenchTmp.json"
exePath = "../build/app/euler.exe"
outPath = "../data/outUnsteady1"

dtBase = 0.125e-1
stepBase = 4000
zeroRecForStepsBase = 0
casePrefix = "GEN3_CylinderB1_RE1200_M01_TS"
orthConfigName = "euler_config.json"
orthRunScriptName = "srunEuler.sh"
orthRunScript = r"""#!/bin/bash
#SBATCH --partition=amd_256
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
export UCX_TLS=sm,self,ud_verbs
export DNDS_ARRAY_STRATEGY_USE_IN_SITU=0
export DNDS_USE_STRONG_SYNC_WAIT=0
export DNDS_USE_ASYNC_ONE_BY_ONE=1
echo "UCX_TLS=${UCX_TLS}"
which mpirun
cp ../../../build/app/euler.exe .
ls -la euler.exe
mpirun euler.exe 1
"""

class JSONWithLineCommentsDecoder(json.JSONDecoder):
    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str) -> any:
        s = '\n'.join(l if not l.lstrip().startswith('//') else '' for l in s.split('\n'))
        return super().decode(s)

fBaseConf = open(baseConfName, "r")
baseConf = json.load(fBaseConf, object_pairs_hook=OrderedDict, cls=JSONWithLineCommentsDecoder)
fBaseConf.close()

print(json.dumps(baseConf, indent=4))




mults = [0.5,1,2,4,8]
ODEs = [("HM3", 401), ("BDF2", 1), ("ESDIRK", 0)]
# ODEs = [("SDIRK", 101)]
writtens = []
for ODE in ODEs:
    for iT in range(1,len(mults) + 1):
        caseName = casePrefix + "%d_%s" % (iT, ODE[0])
        mult = mults[iT - 1]
        confCur = copy.deepcopy(baseConf)
        confCur["timeMarchControl"]["dtImplicit"] = float(dtBase * mult)
        confCur["timeMarchControl"]["nTimeStep"] = int(round(stepBase / mult))
        confCur["timeMarchControl"]["odeCode"] = int(ODE[1])
        confCur["implicitReconstructionControl"]["zeroRecForSteps"] = int(round(zeroRecForStepsBase / mult))
        
        outDir = os.path.join(outPath, caseName)
        if not forceWrite:
            if ignoreExist and os.path.exists(outDir):
                continue
            assert not os.path.exists(outDir)
        
        os.makedirs(outDir, exist_ok=True)
        with open(os.path.join(outDir, orthConfigName), "w+") as fout:
            json.dump(confCur, fout, indent=4)
        with open(os.path.join(outDir, orthRunScriptName), "w+") as fout:
            fout.write(orthRunScript)
        writtens.append(caseName)

print("Written: %d" %(len(writtens)))
print(writtens)
        
        

