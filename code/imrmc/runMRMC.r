library(iMRMChongfei)

args = commandArgs(trailingOnly=TRUE)
print(args[1])
print(args[2])

filename <- args[1]
df <- read.csv(filename,skip=8, header=FALSE)
names(df) <- c('readerID','caseID','modalityID','score')

result <- doIMRMC_R(df)

write.csv(result$Ustat,file=args[2])
print('done')