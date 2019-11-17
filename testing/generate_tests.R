# cmd line params
ARGS <- commandArgs(trailingOnly=TRUE)
INSTFILE <- ARGS[1]
INSTFOLDER <- ARGS[2]
OUTFOLDER <- ARGS[3]
MYEXE1 <- ARGS[4]
MYEXE2 <- ARGS[5]
GOEVAL <- ARGS[6]
SECSPERCONT <- 2L

# define printf (some versions of R don't have it)
printf <- function (..., file = "", sep = " ", fill = FALSE, labels = NULL, append = FALSE)
	cat(sprintf(...), file = file, sep = sep, fill = fill, labels = labels, append = append)

# function for finding file in a directory or below

find_dir_down <- function(startdir, dirname){
	alldirsdown <- list.dirs(startdir)
	idx <- grep(dirname, alldirsdown, fixed=TRUE)[1]
	if(is.na(idx)) return(NA)
	return(alldirsdown[idx])
}

# function for finding file in a directory or above

find_file_up <- function(startdir, fname, maxup=3L){
	up <- 0L
	while(up <= maxup){
		flocation <- paste0(startdir, "/", fname)
		if(file.exists(flocation)) return(flocation)
		startdir <- paste0(startdir, "/..")
		up <- up + 1
	}
	return(NA)
}

# function to find instance files

instance_files <- function(network, scenario){
	files <- paste0("case.", c("con", "inl", "raw", "rop"))
	instancedir <- find_dir_down(INSTFOLDER, paste0(network, "/scenario_", scenario))
	flocs <- character(length(files))
	for(fix in 1:length(files)){
		flocs[fix] <- find_file_up(instancedir, files[fix])
	}
	return(list(con=flocs[1], inl=flocs[2], raw=flocs[3], rop=flocs[4]))
}

# function to get number of contingencies

numcontingencies <- function(confile){
	res <- system(paste0("wc -l ", confile), intern=TRUE)
	nlines <- as.integer(strsplit(res, " ")[[1]][1])
	return((nlines-1L)%/%3L)
}

# function to compute express time in seconds as hh:mm:ss

secs2hms <- function(seconds){
	hours <- seconds%/%3600L
	seconds <- seconds - hours*3600L
	minutes <- seconds%/%60L
	seconds <- as.integer(seconds - minutes*60L)
	return(sprintf("%02d:%02d:%02d", hours, minutes, seconds))
}

# functions to write submission files

write_submit_myexe1 <- function(fname, division, jobname){
	if(division=="R"){
		timelimitsec <- 600L
		divcode <- 1L
	}else{
		timelimitsec <- 2700L
		divcode <- 2L
	}
	timelimithms <- secs2hms(timelimitsec)
	con <- file(fname, "w")
	printf("#!/bin/sh\n#SBATCH --account=gridopt\n#SBATCH --partition=pbatch\n#SBATCH --nodes=4\n#SBATCH --tasks-per-node=36\n", file=con)
	printf("#SBATCH --time=%s\n#SBATCH --job-name=%s\n\n", timelimithms, jobname, file=con)
	printf("module load intel/19.0.4\nmodule load mkl/2019.0\nmodule load impi/2019.0\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/tce/packages/impi/impi-2019.0-intel-19.0.4/lib/release\n\n", file=con)
	printf("srun %s case.con case.inl case.raw case.rop %d %d SomeNetwork", MYEXE1, timelimitsec, divcode, file=con)
	close(con)
}

write_submit_myexe2 <- function(fname, confile, division, jobname){
	ncon <- numcontingencies(confile)
	timelimitsec <- SECSPERCONT*ncon-100L
	timelimithms <- secs2hms(timelimitsec)
	if(division=="R"){
		divcode <- 1L
	}else{
		divcode <- 2L
	}
	con <- file(fname, "w")
	printf("#!/bin/sh\n#SBATCH --account=gridopt\n#SBATCH --partition=pbatch\n#SBATCH --nodes=4\n#SBATCH --tasks-per-node=36\n", file=con)
	printf("#SBATCH --time=%s\n#SBATCH --job-name=%s\n\n", timelimithms, jobname, file=con)
	printf("module load intel/19.0.4\nmodule load mkl/2019.0\nmodule load impi/2019.0\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/tce/packages/impi/impi-2019.0-intel-19.0.4/lib/release\n\n", file=con)
	printf("srun %s case.con case.inl case.raw case.rop %d %d SomeNetwork", MYEXE2, timelimitsec, divcode, file=con)
	close(con)
}

write_submit_goeval <- function(fname, jobname){
	con <- file(fname, "w")
	printf("#!/bin/sh\n#SBATCH --account=gridopt\n#SBATCH --partition=pdebug\n#SBATCH --nodes=1\n#SBATCH --tasks-per-node=1\n", file=con)
	printf("#SBATCH --time=00:30:00\n#SBATCH --job-name=%s\n\n", jobname, file=con)
	printf("python %s case.raw case.rop case.con case.inl solution1.txt solution2.txt summary.csv detail.csv", GOEVAL, file=con)
	close(con)
}

write_submit_instance <- function(fname, submit_myexe1_fname, submit_myexe2_fname, submit_goeval_fname){
	con <- file(fname, "w")
	printf("#!/bin/sh\n\n", file=con)
	printf("jidmyexe1=$(sbatch %s)\n", submit_myexe1_fname, file=con)
	printf("jidmyexe2=$(sbatch --dependency=afterany:${jidmyexe1##* } %s)\n", submit_myexe2_fname, file=con)
	printf("jidgoeval=$(sbatch --dependency=afterany:${jidmyexe2##* } %s)\n", submit_goeval_fname, file=con)
	printf("printf \"MyExe1=%%s MyExe2=%%s GoEval=%%s\\n\" ${jidmyexe1##* } ${jidmyexe2##* } ${jidgoeval##* }\n", file=con)
	close(con)
	Sys.chmod(fname, mode="700")
}

# function to create test cases

create_test <- function(network, scenario, division){
	insfiles <- instance_files(network, scenario)
	testdirname <- paste0(OUTFOLDER, "/", network, "_scenario_", scenario)
	dir.create(testdirname)
	file.symlink(insfiles$con, paste0(testdirname, "/case.con"))
	file.symlink(insfiles$inl, paste0(testdirname, "/case.inl"))
	file.symlink(insfiles$raw, paste0(testdirname, "/case.raw"))
	file.symlink(insfiles$rop, paste0(testdirname, "/case.rop"))
	write_submit_myexe1(paste0(testdirname, "/submit_myexe1.sh"), division,
		paste0(network, "_scen_", scenario, "_myexe1"))
	write_submit_myexe2(paste0(testdirname, "/submit_myexe2.sh"), insfiles$con, division,
		paste0(network, "_scen_", scenario, "_myexe2"))
	write_submit_goeval(paste0(testdirname, "/submit_goeval.sh"),
		paste0(network, "_scen_", scenario, "_goeval"))
	write_submit_instance(paste0(testdirname, "/submit_instance.sh"),
		"submit_myexe1.sh", "submit_myexe2.sh", "submit_goeval.sh")
}

# read instances file and create test instances

instances <- read.csv(INSTFILE, stringsAsFactors=FALSE)
if(!dir.exists(OUTFOLDER)){
	dir.create(OUTFOLDER, recursive=TRUE)
}
for(i in 1:nrow(instances)){
	with(instances, printf("%s %s ... ", network[i], as.character(scenario[i])))
	with(instances, create_test(network[i], scenario[i], division[i]))
	printf("done.\n")
}
