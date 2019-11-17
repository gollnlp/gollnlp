## Generating many evaluation runs at the same time

R scripts to generate multiple Slurm submissions for executing GO Competition simulations.

### Generating instances

From terminal issue

```
Rscript generate_tests.R INSTFILE INSTFOLDER OUTFOLDER MYEXE1 MYEXE2 GOEVAL
```

where:

- ``INSTFILE`` indicates the CSV file containing the instances to run (see ``trial3_instances.csv`` for an example). Absolute or relative paths allowed.
- ``INSTFOLDER`` indicates the folder containing the instances data (e.g. ``/path/to/goinstances/trial3/``). Absolute paths only. The script will create symlinks to the data files of each instance to run.
- ``OUTFOLDER`` indicates the folder where the simulation will take place. To be created if it does not exists. Absolute or relative paths.
- ``MYEXE1`` MyExe1 executable file. Absolute path.
- ``MYEXE2`` MyExe2 executable file. Absolute path.
- ``GOEVAL`` indicates path to the scoring script of the competition ``test.py``. Absolute path. 

The script will create a separate folder for each instance to run, within ``OUTFOLDER``. Each folder will contain symlinks to the instance data and Slurm submission files for executing MyExe1, MyExe2 and the evaluation script, along with a ``submit_instance.sh`` script. The latter will submit MyExe1, MyExe2 and the evaluation script to Slurm, with the adequate dependencies.

**NOTE:** You can convert any relative path ``../my/relative/path`` into an absolute path as ``$(pwd)/../my/relative/path``.

### Executing instances

To submit all instances you have created to Slurm issue the following commands:

```
cd OUTFOLDER
find . -type d -name "Network*" -exec sh -c "cd {} && ./submit_instance.sh && cd .." \;
```

### Expediting instances

To expedite a particular job that is in the Slurm queue, you will need to identify its ``ID``, and then issue the command

```
scontrol update jobid=ID qos=expedite
```

### Construction of summary file

From terminal issue

```
Rscript build_summary.R RESFOLDER SUMMARYFILE
```

where:

- ``RESFOLDER`` corresponds to the folder containing all the results (usually, equal to ``OUTFOLDER`` above). Absolute or relative path.
- ``SUMMARYFILE`` corresponds to the CSV file where the summary will be written to. Absolute or relative path.
