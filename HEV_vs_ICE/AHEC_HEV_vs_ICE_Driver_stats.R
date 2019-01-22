setwd('C:/Users/tangk/Python/')
source("read_data.R")
source("helper.R")
source("stat_tests.R")
library(jsonlite)


# set directory and load JSON file
directory <- 'P:/Data Analysis/Projects/AHEC EV'
params <- fromJSON(file.path(directory,'params.json'))
params <- clean_json(params)

# check which data needs to be read and read the appropriate data
datasets <- unique(params$test$data)
data = list()
for (d in datasets){
	# read data
	if (params$data[[d]][2]=='orig_ts'){
		data[[d]] <- arrange_bych(read_merged(file.path(directory,'Data'),unique(unlist(params$cat))),
					   channels=params$channels,cutoff=params$cutoff)
	} else if (params$data[[d]][2]=='stat') {
		data[[d]] <- read.csv(file.path(directory,params$data[[d]][1]),row.names=1)
	} else if (params$data[[d]][2]=='ts' & is.character(params$data[[d]][1])){
		data[[d]] <- read.csv(file.path(directory,params$data[[d]][1]))
	} else if (params$data[[d]][2]=='ts' & !is.character(params$data[[d]][1])){
		data[[d]] <- params$data[[d]][1]
	} else {
		print('Unknown data type!')
	}
}

# do each test 
label <- data.frame()
res <- data.frame()
for(i in 1:(dim(params$test)[1])){
	print(paste('Test',i,':',params$test$name[i]))
	test1 <- unlist(params$cat[params$test$test1[i]],use.names=FALSE)
	test2 <- unlist(params$cat[params$test$test2[i]],use.names=FALSE)
	paired <- params$test$paired[i]
	testname <- params$test$testname[i]
	d <- params$test$data[i]
	correct <- params$test$correct[i]
	args <- eval(parse(text=paste('list(', params$test$args[i], ')'))) 
	
	# do test
	if(params$data[[d]][2]=='orig_ts'){
		for(ch in params$channels){
			print(paste('Test ',i,':',ch))
			tc1 <- intersect(names(data[[d]][[ch]]),test1)
			tc2 <- intersect(names(data[[d]][[ch]]),test2)
			
			if(paired){
				j <- intersect(match(tc2,test2),match(tc1,test1))
				tc1 <- test1[j]
				tc2 <- test2[j]
			}	
		
			x <- data[[d]][[ch]][,tc1]
			y <- data[[d]][[ch]][,tc2]
			out <- two.sample.ts(x,y,testname,paired=paired,args=args)
			if(length(res)==0){res <- out$stats
			} else {res <- cbind(res,out$stats)}
			nrep <- dim(out$stats)[2]
			
			name <- data.frame(name=params$test$name[i],
						 test1=params$test$test1[i],
						 test2=params$test$test2[i],
						 test=out$testname,
						 paired=paired,
						 channel=substring(ch,2),
						 alpha=out$alpha)
			name <- do.call("rbind",replicate(nrep,name,simplify=FALSE))
			name[['value']] <- names(out$stats)
			label <- rbind(label,name)
			flush.console()
		}
	} else if(params$data[[d]][2]=='stat'){
		x <- data[[d]][test1,]
		y <- data[[d]][test2,]	
		out <- two.sample.byname(x,y,testname,paired=paired,args=args,vector.form=TRUE)	
		if(length(res)==0){
			res <- out$stats
			nrep <- length(out$stats)
		} else if (is.vector(res)){
			res <- c(res,out$stats)
			nrep <- length(out$stats)
		} else {
			n <- names(out$stats)
			out$stats <- data.frame(rbind(out$stats,replicate(length(out$stats),rep(NaN,dim(res)[1]-1))))
			names(out$stats) <- n
			res <- cbind(res,out$stats)
			nrep <- dim(out$stats)[2]
		}
		name <- data.frame(name=params$test$name[i],
					 test1=params$test$test1[i],
					 test2=params$test$test2[i],
					 test=out$testname,
					 paired=paired,
					 alpha=out$alpha)
		name <- do.call("rbind",replicate(nrep,name,simplify=FALSE))
		name$channel <- as.vector(t(replicate(length(unique(names(out$stats))),sapply(names(x),function(x) substring(x,2),USE.NAMES=FALSE))))
		name$value <- names(out$stats)
		label <- rbind(label,name)
	} #else if(params$data[[d]][2]=='ts'){}
	flush.console()
}


# write to json
params$stats_label <- label
params$stats_path <- 'Rstats.csv'
if(!is.null(params$channels)){params$channels <- sapply(params$channels,function(x) substring(x,2))}
write_json(params,file.path(directory,'params.json'))

# write data to csv (for now. maybe change to feather)
write.csv(res,file.path(directory,'Rstats.csv'))
