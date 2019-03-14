setwd('P:/Data Analysis/Projects/AHEC EV')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/helper.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/stat_tests.R")
library(jsonlite)
library(lars)

directory <- 'P:/Data Analysis/Projects/AHEC EV'
table <- read_table(directory)
features <- read.csv(file.path(directory,'features.csv'))
if('X' %in% names(features)){
  rownames(features) <- features$X
  features <- features[,-1]
}


# calculate differences for Series 1
#subset <- table.query(table, c('Series_1==1', 'Type==\'ICE\'', 'ID11==\'TH\'','Pair_name!=\'SOUL\''))
subset <- table.query(table, c('Series_1==1', 'Type==\'ICE\'', 'ID11==\'TH\''))
ice <- as.vector(na.omit(subset$TC))
ev <- as.vector(na.omit(subset$Pair))
diff.features <- features[ice,]/features[ev, ]
model <- lm(Max_11CHST003STHACRC ~ Ratio_weight, data=diff.features)
summary(model) 
plot(diff.features$Ratio_weight, diff.features$Max_11CHST003STHACRC)


subset <- table.query(table, c('Type==\'ICE\''))
data <- features[subset$TC, ]
responses <- c('Max_11HEAD003STHACRA', 'Max_11CHST003SH3ACRC', 'Max_11CHST003STHACRC')
for (r in responses){
  #model <- lm(Max_11HEAD003STHACRA.partner_weight ~ Weight, data=features)
  model <- eval(parse(text=paste0('lm(', r, ' ~ Weight, data=data)')))
  print(r)
  #print(summary(model)$coefficients['Weight','Estimate']*1e5)
  print(summary(model)$coefficients[,'Pr(>|t|)'])
  print(summary(model)$r.squared)
  print('\n')}


plot(features$Ratio_weight, features$Max_11CHST003STHACRC.partner_weight)
