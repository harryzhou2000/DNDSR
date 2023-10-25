# Notes

## on periodic boundary

for simplicity, the single-block mesh supports 1-to-1 definition of periodic bc, which is, the "main" and "donor" are the same bnd meshes given the rigid body transformation.

periodic donor and main must be non-adjacent (not sharing nodes) (main-donor node mapping each pair is not identical) [*reasoning: if periodicity needs this, means singularity point, replace this point with a small face with sym condition*]

then, record cell2nodePi, and face2nodePi, to augment cell2node ..., and:

`nodePbi` (periodic bits info): bit1-bit2-bit3, means [if peri1][if peri2][if peri3], stored in uint8_t now; the bits show if the node (for the elem) is transferred from a periodic group. if 1-1-1, then p->transBack_1->transBack_2->transBack_3->p_current, and to obtain the geometry-well coordinate p, use the reverse transformation. see Geom::UnstructuredMesh::Deduplicate1to1Periodic() and Geom::PeriodicInfo::GetCoordByBits().

in the sense of peri-duplicated (original geometry) mesh, node in a cell is different when both cell2node(iCell,ic2n) and cell2nodePI(iCell,ic2n) are different; so use both cell2node and cell2nodePi for inner-cell coord calculation

in the sense of de-duplicated mesh, cell2node(iCell,ic2n) is unique, and points to a coord;

to get interpolate face, the faces are the same when both are true:

- face2node are the same,
- face2nodePi are collaborating: the xor_s of (face2nodePbiL_if2n, face2nodePbiR_if2n) are the same

face2nodePi are got from the first cell, like face2node
faceAtr are got from the first cell, (for de-duplicated faces, could be periodic-donor or main)

to query face-coords in cell, check is is periodic, if donor, face2cell[1] needs the face to trans-back; if main, face2cell[1] needs the face to trans; face2cell[0] is always good

to query other cell-coord through face, if face is periodic main, if cell is face2cell[0] then other cell needs trans back; cell is face2cell[1] then other cell needs trans


## Point communication race:

Point-to-point comm in MPI could potentially **cause comm race** when the comm pattern (of a collective-like comm call) is very dense (close to a all_to_all), while the machine topology is sparse (like in a supercomputer).

calling like:

```c
MPI_Request* sendReqs, recvReqs;
//traversing all ranks in comm, which is the densest, actual code would omit those with zero size
for(int rankOther = 0; rankOther < commSize; rankOther++) 
    MPI_SendInit(buf,count,...,rankOher,..., sendReqs + rankOther);
for(int rankOther = 0; recvOther < commSize; rankOther++)
    MPI_RecvInit(buf,count,...,rankdOher,..., recvReqs + rankOther); 
MPI_Startall(commSize, sendReqs);
MPI_Startall(commSize, recvReqs);

MPI_Waitall(commSize, sendReqs, ...);
MPI_Waitall(commSize, recvReqs, ...);
```

would launch multiple data transferring tasks in MPI simultaneously, which is ok on a dense machine like a NUMA machine, but it causes network racing and saturation in a sparse machine like a common multi-node supercomputer. The best practice, for the performance of communication, would be like a MPI_Alltoall, which refers to the current machine's topology. On the other hand, alltoall is always global and brings extra zero-sized-data communication overhead globally, which is O(np) overhead. So, DNDS assumes that the comm pattern is sparse enough, and when the the sparsity exceeds that of the machine, the current implementation of DNDS uses one-by-one send/recv, like: 

```c
MPI_Request* sendReqs, recvReqs;
//traversing all ranks in comm, which is the densest, actual code would omit those with zero size
for(int rankOther = 0; rankOther < commSize; rankOther++) 
    MPI_SendInit(buf,count,...,rankOher,..., sendReqs + rankOther);
for(int rankOther = 0; recvOther < commSize; rankOther++)
    MPI_RecvInit(buf,count,...,rankdOher,..., recvReqs + rankOther); 
MPI_Startall(commSize, sendReqs);

// foreach_recvReqs: start,wait
for(int rankOther = 0; recvOther < commSize; rankOther++)
{
    MPI_Start(recvReq[rankOther]);
    MPI_Wait(recvReq[rankOther]);
}
// 
MPI_Waitall(commSize, sendReqs, ...);
```

which is receiving one-by-one. Sending one-by-one is also ok for solving network race. 

Actually, chunk-by-chunk should be used on a sparse machine, with the knowledge of local network capacity. But since sparse decomposed 3-d meshes generally have very small communication overhead, optimization on this is delayed.


### Dynamic-Reforming: Loss of Performance

For DLR case used for eulerSA3D in `d082620525d1d9a07889c9e8c1a9bede70ebe236`, on GS machine, times is: 2.1582/it

If using nVars_Fixed = -1  (original 6), time is: 3.3944/1t


### Note on HM3: 

must use residual instead of uinc for convergence monitoring!!!

### HM3 testing:

237m14.899s for hm3 run on HZAU
121m49.267s for ESDIRK4



