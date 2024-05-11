# Caveats for IGRINS2 reduction

## Readout pattern removal

- Readout pattern removal in IGRINS2 can be currently unstable, largely due to issues with reference pixel correction.
- For 'FLAT', we changed the default removal option to "global_median".
  - "--rp-remove-mode=0" is not necessary anymore.
- Please investigate the debug image for any readout pattern issues.

## Vertical pattern

- IGRINS2 also suffer from vertical pattern.
- A method to minimize this pattern is still in progress.
