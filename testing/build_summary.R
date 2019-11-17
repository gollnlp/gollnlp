# cmd line params
ARGS <- commandArgs(trailingOnly=TRUE)
RESFOLDER <- ARGS[1]
SUMMARYFILE <- ARGS[2]

# define printf (some versions of R don't have it)
printf <- function (..., file = "", sep = " ", fill = FALSE, labels = NULL, append = FALSE)
	cat(sprintf(...), file = file, sep = sep, fill = fill, labels = labels, append = append)

# function to find log files

log_files <- function(exedir){
	files <- sort(list.files(exedir, pattern="*.out", full.names=TRUE))
	return(list(myexe1=files[1], myexe2=files[2], goeval=files[3]))
}

# function to get myexe2 execution time

myexe2_time <- function(myexe2log){
	grepres <- system(paste0("grep \"\\-\\-finished in\" ", myexe2log), intern=TRUE)
	grepres <- strsplit(grepres, " ")[[1]]
	return(as.numeric(grepres[length(grepres)-1L]))
}

# function to get goeval stats

goeval_stats <- function(goevallog){
	goevaldata <- readLines(goevallog)
	objpos <- grep("obj: ", goevaldata)
	goevaldata <- goevaldata[objpos:(objpos+5)]
	goevaldata <- strsplit(goevaldata, ": ")
	return(list(
		obj=as.numeric(goevaldata[[1]][2]),
		cost=as.numeric(goevaldata[[2]][2]),
		penalty=as.numeric(goevaldata[[3]][2]),
		max_obj_viol=as.numeric(goevaldata[[4]][2]),
		max_nonobj_viol=as.numeric(goevaldata[[5]][2]),
		infeas=as.integer(goevaldata[[6]][2])))
}

# function to get network name and scenario number

network_scenario <- function(exedir){
	netscen <- strsplit(tail(strsplit(exedir, "/")[[1]], 1), "_scenario_")[[1]]
	return(list(network=netscen[1], scenario=as.integer(netscen[2])))
}

# function to generate a single-row data frame summary

single_row_df <- function(exedir){
	netscen <- network_scenario(exedir)
	logs <- log_files(exedir)
	me2time <- myexe2_time(logs$myexe2)
	gostats <- goeval_stats(logs$goeval)
	df <- data.frame(
		Folder=exedir,
		Network=netscen$network,
		Scenario=netscen$scenario,
		Objective=gostats$obj,
		Cost=gostats$cost,
		Penalty=gostats$penalty,
		Max_Obj_Violation=gostats$max_obj_viol,
		Max_NonObj_Violation=gostats$max_nonobj_viol,
		Infeasible=gostats$infeas,
		MyExe2_Time=me2time,
		detail=paste0(exedir, "/detail.csv"),
		MyExe1_log=logs$myexe1,
		MyExe2_log=logs$myexe2,
		solution1=paste0(exedir, "/solution1.txt"),
		solutionpd=paste0(exedir, "/solution_b_pd.txt"),
		solution2=paste0(exedir, "/solution2.txt"),
		stringsAsFactors=FALSE)
	return(df)
}

single_row_df_err <- function(exedir){
	df <- data.frame(
		Folder=exedir,
		Network="",
		Scenario=0,
		Objective=Inf,
		Cost=NA,
		Penalty=NA,
		Max_Obj_Violation=NA,
		Max_NonObj_Violation=NA,
		Infeasible=NA,
		MyExe2_Time=NA,
		detail=NA,
		MyExe1_log=NA,
		MyExe2_log=NA,
		solution1=NA,
		solutionpd=NA,
		solution2=NA,
		stringsAsFactors=FALSE)
	return(df)
}

# loop over all directories in the result folder

exedirs <- list.dirs(RESFOLDER, recursive=FALSE)
summarydf <- data.frame()
for(d in exedirs){
	printf("Processing results at %s ... ", d)
	summdfd <- tryCatch(
		single_row_df(d),
		error=function(error_message){
			message(paste0("Problem detected while processing ", d, ". R error message:"))
			message(error_message)
			return(single_row_df_err(d))
			}
		)
	summarydf <- rbind(summarydf, summdfd)
	printf("done.\n")
}

# complete directories with working dir information

summarydf$Folder <- paste0(getwd(), "/", summarydf$Folder)
summarydf$detail <- paste0(getwd(), "/", summarydf$detail)
summarydf$MyExe1_log <- paste0(getwd(), "/", summarydf$MyExe1_log)
summarydf$MyExe2_log <- paste0(getwd(), "/", summarydf$MyExe2_log)
summarydf$solution1 <- paste0(getwd(), "/", summarydf$solution1)
summarydf$solutionpd <- paste0(getwd(), "/", summarydf$solutionpd)
summarydf$solution2 <- paste0(getwd(), "/", summarydf$solution2)

# write summary df

write.table(summarydf, SUMMARYFILE, row.names=FALSE, sep=",")
