setwd('C:/Users/tangk/pmg-projects/AHEC/HEV_vs_ICE')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/helper.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/PMG/COM/stat_tests.R")
library(jsonlite)

# set directory and load JSON file
directory <- 'P:/Data Analysis/Projects/AHEC EV'
params <- fromJSON(file.path(directory,'params.json'), simplifyDataFrame=FALSE)
tests <- rbind_pages(lapply(params$test, data.frame))
tests <- data.frame(as.matrix(tests), stringsAsFactors = FALSE)
data.files <- unlist(params$data)

# check which data needs to be read and read the appropriate data
datasets <- unique(tests$data)
data = list()
for (d in datasets){
  data[[d]] <- read.csv(file.path(directory, data.files[[d]]), stringsAsFactors = FALSE)
  if('X' %in% names(data[[d]])){
    rownames(data[[d]]) <- data[[d]]$X
    data[[d]] <- data[[d]][,-1]
  }
}

# do each test 
for(i in 1:(dim(tests)[1])){
	print(paste('Test',i,':',tests[i, 'name']))
	test1 <- unlist(params$cat[tests[i, 'test1']],use.names=FALSE)
	test2 <- unlist(params$cat[tests[i, 'test2']],use.names=FALSE)
	paired <- as.logical(tests[i, 'paired'])
	testname <- tests$testname[i]
	d <- tests$data[i]
	args <- eval(parse(text=paste('list(', tests$args[i], ')'))) 
	x <- data[[d]][test1,] 
	y <- data[[d]][test2,]
	out <- two.sample.byname(x, y, testname, paired=paired, args=args, vector.form=TRUE)
	params$test[[i]]$res <- as.vector(out$stats)
	nlabel <- length(unique(names(out$stats)))
	nrep <- length(names(out$stats))/nlabel
	params$test[[i]]$label1 <- rep(names(x), each=nlabel)
	params$test[[i]]$label2 <- rep(names(out$stats)[1:nlabel], nrep)
	flush.console()
}

# write to json
write_json(params,file.path(directory,'params.json'))
