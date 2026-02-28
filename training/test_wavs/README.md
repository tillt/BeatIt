# Test WAV Canonical Notes

Each file in `training/` has a corresponding spec file here. Fill in the
canonical tempo and the sample-frame index of the first downbeat.

Use these files as the single source of truth when evaluating refiner
changes or model updates.

Additional rule for current sparse-window notes:

- Files with long weak or ambient breaks should document rhythmic-section checks
  separately from generic interior-window drift checks.
- Do not treat low-onset break material as a canonical phase-alignment region
  just because it lies near the geometric middle of the file.
