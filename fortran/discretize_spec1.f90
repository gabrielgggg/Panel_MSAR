PROGRAM discretize_spec1
  ! Spec 1, 2026-09-17 SA panel, full sample.
  ! Stationary MS-AR for z. Rho is 0.99 in every regime.
  USE NL, ONLY: wp, discretizeMSAR
  IMPLICIT NONE

  INTEGER, PARAMETER :: nreg = 3
  INTEGER, PARAMETER :: nn   = 15
  REAL(wp), PARAMETER :: nsds = 3.0_wp
  REAL(wp), PARAMETER :: rrho = 0.99_wp

  REAL(wp), DIMENSION(nreg) :: mus, rrhos, sstds
  REAL(wp), DIMENSION(nreg, nreg) :: Pi
  REAL(wp), DIMENSION(nn) :: zgrid
  REAL(wp), DIMENSION(nreg*nn, nreg*nn) :: bigTran
  REAL(wp), DIMENSION(nreg*nn) :: stationary
  INTEGER, DIMENSION(nreg*nn, 3) :: mmap
  INTEGER, DIMENSION(nreg, nn) :: revmap

  ! Regime order is increasing mu. Median mu is pinned at 0.
  mus   = (/ -0.2514_wp, 0.0_wp, 1.7443_wp /)
  rrhos = rrho
  sstds = (/  0.0699_wp, 0.0161_wp, 0.0090_wp /)

  ! Rows from, columns to. Entries are counts per 10000; each row adds to 10000.
  Pi = TRANSPOSE(RESHAPE( [ &
    7351.0_wp, 1931.0_wp,  718.0_wp, &
     650.0_wp, 9300.0_wp,   50.0_wp, &
     280.0_wp,   23.0_wp, 9697.0_wp  &
  ], [nreg, nreg] )) / 10000.0_wp

  CALL discretizeMSAR(mus, rrhos, sstds, Pi, nn, nsds, &
    zgrid, bigTran, mmap, revmap, stationary)

END PROGRAM discretize_spec1
