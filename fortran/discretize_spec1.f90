PROGRAM discretize_spec1
  ! RE intercepts, common linear trend, E[z]=0.
  ! 2026-09-17 SA panel, full sample. Stationary MS-AR for z only.
  USE NL, ONLY: wp, discretizeMSAR
  IMPLICIT NONE

  INTEGER, PARAMETER :: nreg = 3
  ! nn is the length of the shared z grid. The joint chain has nreg*nn states.
  INTEGER, PARAMETER :: nn   = 7
  REAL(wp), PARAMETER :: nsds = 3.0_wp

  REAL(wp), DIMENSION(nreg) :: mus, rrhos, sstds
  REAL(wp), DIMENSION(nreg, nreg) :: Pi
  REAL(wp), DIMENSION(nn) :: zgrid
  REAL(wp), DIMENSION(nreg*nn, nreg*nn) :: bigTran
  REAL(wp), DIMENSION(nreg*nn) :: stationary
  INTEGER, DIMENSION(nreg*nn, 3) :: mmap
  INTEGER, DIMENSION(nreg, nn) :: revmap

  ! Regime order is increasing mu. E[z]=0 is already imposed.
  mus   = (/ -1.0703_wp, -0.0794_wp, 0.2903_wp /)
  rrhos = (/  0.9899_wp,  0.9900_wp, 0.9899_wp /)
  sstds = (/  0.0738_wp,  0.0186_wp, 0.0069_wp /)

  ! Rows from, columns to. Entries are counts per 10000; each row adds to 10000.
  Pi = TRANSPOSE(RESHAPE( [ &
    6943.0_wp, 2876.0_wp,  181.0_wp, &
     638.0_wp, 8778.0_wp,  584.0_wp, &
      52.0_wp,  506.0_wp, 9442.0_wp  &
  ], [nreg, nreg] )) / 10000.0_wp

  CALL discretizeMSAR(mus, rrhos, sstds, Pi, nn, nsds, &
    zgrid, bigTran, mmap, revmap, stationary)

END PROGRAM discretize_spec1
