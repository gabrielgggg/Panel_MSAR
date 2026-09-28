PROGRAM discretize_grids_example
  ! 1985Q1-2026Q2 cycle. Within-country slope, E[z]=0.
  ! Each regime has its own z grid. The state is the node on the
  ! concatenated grid; the regime is the block that node sits in.
  USE NL, ONLY: wp, discretizeMSARgrids
  IMPLICIT NONE

  INTEGER, PARAMETER :: nreg = 3
  INTEGER, PARAMETER :: noZ  = 5
  REAL(wp), PARAMETER :: nsd = 2.0_wp
  INTEGER, PARAMETER :: nstate = nreg * noZ

  REAL(wp), DIMENSION(nreg) :: mus, rrhos, sstds
  REAL(wp), DIMENSION(nreg, nreg) :: Pi
  REAL(wp), DIMENSION(nstate) :: zgrid, stationary
  REAL(wp), DIMENSION(nstate, nstate) :: bigTran
  INTEGER :: i, is

  ! Regime order is increasing mu.
  mus   = (/ -0.7737_wp, -0.7471_wp, 1.1467_wp /)
  rrhos = (/  0.9900_wp,  0.9900_wp, 0.9900_wp /)
  sstds = (/  0.0733_wp,  0.0159_wp, 0.0098_wp /)

  ! Rows from, columns to. Counts per 10000; each row adds to 10000.
  Pi = TRANSPOSE(RESHAPE( [ &
    7223.0_wp, 2000.0_wp,  777.0_wp, &
     610.0_wp, 9385.0_wp,    5.0_wp, &
     284.0_wp,    0.0_wp, 9716.0_wp  &
  ], [nreg, nreg] )) / 10000.0_wp

  CALL discretizeMSARgrids(mus, rrhos, sstds, Pi, noZ, nsd, &
    zgrid, bigTran, stationary)

  WRITE(*,*) 'node  regime         z'
  DO i = 1,nstate
    is = (i - 1) / noZ + 1
    WRITE(*,'(I4, I8, F14.6)') i, is, zgrid(i)
  END DO

  WRITE(*,*)
  WRITE(*,*) 'P(to | from), one row per origin node'
  DO i = 1,nstate
    WRITE(*,'(I4, 15F8.4)') i, bigTran(i, :)
    WRITE(*,'(A, F12.6)') '  row sum', SUM(bigTran(i, :))
  END DO

  WRITE(*,*)
  WRITE(*,*) 'stationary'
  WRITE(*,'(15F8.4)') stationary
END PROGRAM discretize_grids_example
