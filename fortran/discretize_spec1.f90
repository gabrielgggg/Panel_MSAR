PROGRAM discretize_spec1
  ! EMBI countries, 1990 onward. Common growth trend removed.
  ! RE intercepts and Barro catch-up are not passed in. E[z]=0.
  ! Stationary MS-AR for z only.
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
  mus   = (/ -0.0845_wp, -0.0161_wp, 0.1099_wp /)
  rrhos = (/  0.9900_wp,  0.7826_wp, 0.9900_wp /)
  sstds = (/  0.0243_wp,  0.0892_wp, 0.0095_wp /)

  ! Rows from, columns to. Entries are counts per 10000; each row adds to 10000.
  ! Row 1's last count is 657 so the row is 10000 (printed 0.0658 rounded up).
  Pi = TRANSPOSE(RESHAPE( [ &
    9023.0_wp,  320.0_wp,  657.0_wp, &
    3017.0_wp, 6803.0_wp,  180.0_wp, &
     594.0_wp,   90.0_wp, 9316.0_wp  &
  ], [nreg, nreg] )) / 10000.0_wp

  CALL discretizeMSAR(mus, rrhos, sstds, Pi, nn, nsds, &
    zgrid, bigTran, mmap, revmap, stationary)

END PROGRAM discretize_spec1
