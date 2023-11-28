import sys
import os
import re
import os.path

prefix = "../CylinderA1_RE2000_BDF2_8000x"
namePrefix = "CylinderA1_RE2000__"
mcrIn = "GetCURL2D.mcr"
mcrOut = "GetCURL2D_Seq.mcr"

names = os.listdir(prefix)

namesNeed = []
for name in names:
    matched = re.match(namePrefix + r"(\d+)\.plt", name)
    if matched:
        namesNeed.append((os.path.join(prefix, name), int(matched.group(1))))


namesNeed = sorted(namesNeed, key=lambda v: v[1])

# print(namesNeed)

mcrInFile = open(mcrIn, "r")
mcrOutFile = open(mcrOut, "w")

lines = mcrInFile.readlines()

for (name, number) in namesNeed:
    mcrTxt = ""
    for line in lines:
        line = line.replace(r"%%READ", os.path.realpath(name))
        line = line.replace(
            r"%%WRITE",
            os.path.realpath(
                os.path.join(prefix, namePrefix + "%05d" % (number) + ".png")))
        mcrTxt += line

    mcrOutFile.write(mcrTxt + "\n\n")
