# Extended Euler rotating-source correction

NS_EX and NS_EX_3D store three absolute fluid-velocity components, including
on 2D geometry, when `USE_ABS_VELO_IN_ROTATION` is enabled. Components are
expressed in rotating Cartesian axes; total energy contains absolute kinetic
energy. Face transport uses fluid velocity relative to the rotating grid.

The variant-based source contributor previously used centrifugal force,
double-Coriolis force and energy work unconditionally. Those formulas belong
to relative velocity and disagree with the enabled state/flux convention.
The contributor now follows the same build convention as fixed-size Euler:
absolute momentum receives `-Omega cross momentum`, its stored Jacobian is
the negative source derivative, and rotation adds no energy source. The
relative-velocity build branch is preserved. Modes 0/1/2 follow the existing
fixed-size source behavior, including the returned momentum source in Mode 2.

`euler_rotating_frame` checks both extended models with scalar-component
formulas and finite differences, nonzero/disabled/zero rotation, off-axis 3D
rotation, and unchanged density/energy/extra scalars. It is registered in the
aggregate Euler test target. This focused correction does not establish
end-to-end rotating reacting-flow or multi-domain validation.
