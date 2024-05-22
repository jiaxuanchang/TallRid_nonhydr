C $Header$
C $Name$

C****
C**** CHIP HEADER FILE
C****

C   SCCS VERSION @(#)sibber.h   1.1 10/15/92

      INTEGER  NTYPS, FRSTCH, MemFac
      INTEGER  NLAY, SFCLY, ROOTLY, RECHLY

      _RL  ZERO, ONE, PIE
      _RL ALHE, ALHS, ALHM, TF, STEFAN, RGAS, SHW, SHI, RHOW, GRAV
      _RL EPSILON, NOSNOW

      PARAMETER (NTYPS = 10, FRSTCH = 1, MemFac = 5)
      PARAMETER (NLAY = 3)
      PARAMETER (SFCLY = 1, ROOTLY = SFCLY + 1, RECHLY = ROOTLY + 1)

      PARAMETER (ZERO = 0., ONE = 1., PIE = 3.14159265)
      PARAMETER (ALHE = 2.4548E6, ALHS = 2.8368E6, ALHM = ALHS-ALHE)
      PARAMETER (TF = 273.16)
      PARAMETER (STEFAN = 5.669E-8)
      PARAMETER (RGAS = .286*1003.5)
      PARAMETER (SHW = 4200., SHI = 2060.)
      PARAMETER (RHOW = 1000.)
      PARAMETER (GRAV = 9.81)
      PARAMETER (EPSILON = 18.01/28.97)
      PARAMETER (NOSNOW = 0.)
