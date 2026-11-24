# Raw validation data

The `validation-data.tar.gz` archive contains the raw validation records and
is available as a release download. [The manifest](validation-data.json) lists
checksums for the archive and the files below.

- Eight files of individual simulation measurements covering Figures 2–7,
  including Figure 4's `replicas.npz` and `validation_replicas.npz`.
- The 133 run records for Figure 7 effective frequency.
- The 80 rows of individual replica values underlying the Figure 4 sampled minimum map.

[Download `validation-data.tar.gz` (301 MB)](https://github.com/AFBurnet/Energetics-of-coupled-stochastic-circular-limit-cycle-oscillators/releases/download/revision-update/validation-data.tar.gz).
Run the following command from the repository root.

```bash
python data-bundles/restore.py /path/to/validation-data.tar.gz
```

The script verifies the archive and file hashes before restoring the records
at their figure paths. Restored raw files are ignored by Git. Figure
regeneration does not require the archive. The archive does not contain the
original Figure 8 production trajectory caches.

The replica replay commands for Figures 2, 3, 5, 6 and 7 rebuild summaries
in their input directory. To keep the distributed summaries unchanged,
copy that figure's `data/` into a separate run directory and validate the
copy. For example, after restoring the archive:

```bash
mkdir -p results
mkdir results/fig02_validation
cp -R figures/fig02_radial_pair/data results/fig02_validation/data
python figures/fig02_radial_pair/figure2.py validate --outdir results/fig02_validation
```

The other figure READMEs identify the corresponding data directories.
Figure 4's `validate` reads the restored measurements without changing its
saved summaries; use `--report FILE` to save the replayed checks.
