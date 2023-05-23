function N = SF_Pyramid5(xi,et,zt)

NB = SF_Quad4(xi, et, 0);
Nz = SF_Line2(zt * 2 - 1, 0, 0);

N = [...
    NB(1) * Nz(1)
    NB(2) * Nz(1)
    NB(3) * Nz(1)
    NB(4) * Nz(1)
    Nz(2)
]';

end