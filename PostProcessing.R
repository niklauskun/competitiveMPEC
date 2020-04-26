rm(list=ls())

install.packages("mvtnorm")
install.packages("emmeans")
install.packages("sjstats")
install.packages("sjPlot")
install.packages('xtable')

library(ggplot2)
library(openxlsx)
library(plyr)
library(reshape2)
library(skimr)
library(dplyr)
#library(sjPlot)
library(table1)
library(xtable)

baseWD <- "C:/Users/Luke/Documents/GitHub/competitiveMPEC"
setwd(paste(baseWD, "basedispatch", sep="/"))

## Load model results ####

# helper function to read all files in date range of a specific output
readFiles <- function(filename, dates, dateWD, subFolder="results"){
  
  for(i in 1:length(dates)){
    date <- dates[i]
    dateString <- paste(as.character(format(date, "%m")), as.character(format(date, "%d")), as.numeric(format(date, "%Y")), sep=".")
    setwd(paste(dateWD, dateString, subFolder, sep="/"))
    
    # load file
    df <- read.csv(filename)
    df$date <- date
    
    if(i == 1){
      fullDF <- df
    } else{
      fullDF <- rbind(fullDF, df)
    }
  }
  return(fullDF)
}

loadResults <- function(dates,folder){
  
  dateMonth <- unique(format(dates, "%b"))  # extract month of dates for directory
  dateYear <- unique(format(dates, "%Y"))  # extract year of dates for directory
  
  if (length(dateMonth) == 1){
    dayStart <- as.numeric(format(dates[1], "%d"))
    dayEnd <- max(as.numeric(format(dates[length(dates)], "%d")),dayStart+6)
    #dayEnd <- dayEnd
    # case options: NoReserves, SimpleORDC, ConstantReserves, withORDC
    # there will be new case options but this should be ok
    #directory <- paste(dateMonth, dayStart, dayEnd, dateYear, case, sep="_")
    dateResultsWD<-paste(baseWD, folder, sep="/")
    
  } else{
    print("Error. Multiple months in date range.")   # multiple months yields error for now, can update based on directory format
    return(NULL)
  }
  
  modelLMP <- readFiles("zonal_prices.csv", dates, dateResultsWD)
  txFlows <- readFiles("tx_flows.csv", dates, dateResultsWD)
  offer <- readFiles("generator_segment_offer.csv", dates, dateResultsWD)
  dispatch <- readFiles("generator_dispatch.csv", dates, dateResultsWD)
  storage <- readFiles("storage_dispatch.csv", dates, dateResultsWD)
  VRE <- readFiles("renewable_generation.csv", dates, dateResultsWD)
  
  #ordc <- readFiles("full_ordc.csv", dates, dateResultsWD, subFolder="inputs")
  gens <- readFiles("PJM_generators_full.csv", dates, dateResultsWD, subFolder="inputs")
  zonalLoad <- readFiles("timepoints_zonal.csv", dates, dateResultsWD, subFolder="inputs")
  emissions <- readFiles("generator_segment_marginalcost.csv", dates, dateResultsWD, subFolder="inputs")
  
  # some formatting
  gens <- gens[!duplicated(gens),] # remove duplicate generators
  gens <- gens[,c("Name", "Zone", "Category")]  # subset generator columns
  
  #reformat zonal prices so it matches previous format
  modelLMP <- modelLMP[,c("X","hour","LMP","date")]
  
  # return results
  results <- list(modelLMP, zonalLoad, dispatch, gens, txFlows, storage, offer, emissions)
  names(results) <- c("modelLMP", "zonalLoad", "dispatch", "gens", "txFlows", "storage", "offer","emissions")
  return(results)
}

# helper function to load data from all four cases
loadAllCases <- function(dates,folder="RTSDispatchCases"){
  results <- loadResults(dates,folder)
  return(results)
}

aggregateCaseData <- function(results, targetData){
  newDF <- data.frame()
  # format results by case
  for(case in names(results)){
    dfTemp <- results[[case]][[targetData]] 
    dfTemp$case <- case
    newDF <- rbind(newDF, dfTemp)
  }
  return(newDF)
}

#zonal prices

plotPrices <- function(results,dates,plotTitle,hours=24){
  prices <- results[['modelLMP']]
  prices$zone <- substr(prices[,1],start=1,stop=1)
  prices$zone <- paste0("Area ",prices$zone)
  prices$busID <- substr(prices[,1],start=2,stop=3)
  prices$datetime <- as.POSIXct(with(prices, paste(date, hour)), format = "%Y-%m-%d %H")
  prices$X <- as.character(prices$X)
  
  #Luke's plotting code (active)
  ggplot(data=prices, aes(x=datetime, y=LMP, color=busID)) + geom_line(lwd=1.5) + 
    facet_wrap(~zone, nrow=1) + 
    theme_classic() + ylab("$/MWh") + xlab("") +
    scale_x_datetime() +
    guides(colour=guide_legend(title="Bus: ", nrow=5))+
    theme(legend.text = element_text(size=24),
          legend.title = element_text(size=30),
          legend.position = "bottom",
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=16),
          axis.text.y= element_text(size=16),
          strip.text.x = element_text(size = 24)) +
    ggtitle(paste("LMP by Bus for", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("prices ", plotTitle, ".png"), width=20, height=8)
  
  return(prices)
}

compareplotPrices <- function(prices_df1,prices_df2){
  pricesdelta_df <- prices_df1[,c("datetime","zone","busID")]
  pricesdelta_df$LMP <- prices_df1$LMP - prices_df2$LMP
  
  #Luke's plotting code (active)
  ggplot(data=pricesdelta_df, aes(x=datetime, y=LMP, color=busID)) + geom_line(lwd=1.5) + 
    facet_wrap(~zone, nrow=1) + 
    theme_classic() + ylab("$/MWh") + xlab("") +
    scale_x_datetime() +
    guides(colour=guide_legend(title="Bus: ", nrow=5))+
    theme(legend.text = element_text(size=24),
          legend.title = element_text(size=30),
          legend.position = "bottom",
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=16),
          axis.text.y= element_text(size=16),
          strip.text.x = element_text(size = 24)) +
    ggtitle(paste("Delta LMP by Bus"))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("prices delta",".png"), width=20, height=8)
  
}

plotDispatch <- function(results, dates, plotTitle, hours=24){
  
  dispatch <- results[["dispatch"]]
  gens <- results[["gens"]]
  
  offer <- results[['offer']]
  #print(offer)
  #offer$SegmentEmissions
  offer$segID <- gsub("[[:print:]]*-", "", offer[,1])
  offer$genID <- gsub("-\\d","",offer[,1])
  offer$area <- substr(offer$Zone,start=1,stop=1)
  
  offer <- merge(offer, gens[,c("Name", "Category")], by.x="genID", by.y="Name", all.x=T)
  #drop duplicated entries in output
  offer <- offer[!duplicated(offer), ]
  # summarize by fuel type
  fuelemissions <- ddply(offer, ~ date + hour + area, summarise, Emissions = sum(SegmentEmissions))
  fuelemissions$area <- factor(fuelemissions$area)
  fuelemissions$datetime <- as.POSIXct(with(fuelemissions, paste(date, hour)), format = "%Y-%m-%d %H")
  
  #then melt into same format as fueldf
  
  # calculate all zones
  all_emissions <- ddply(fuelemissions, ~ datetime, summarise, Emissions = sum(Emissions))
  all_emissions$area <- "All Zones"
  
  fuelemissions <- fuelemissions[,c("datetime", "Emissions", "area")]
  fuelemissions <- rbind(fuelemissions, all_emissions)
  colnames(fuelemissions) <- c("datetime","Emissions","zone")
  fuelemissions$zone <- as.factor(fuelemissions$zone)
  fuelemissions$Category <- "Coal"
  #print(fuelemissions)
  
  # subset dispatch output to single day (include columns for date and case as well)
  dispatch <- dispatch[, c(1:(hours+1), dim(dispatch)[2])]
  colnames(dispatch) <- c("id", 0:(hours-1), "date")
  dispatch$zone <- substr(dispatch[,1],start=1,stop=1)
  dispatch$plant <- gsub("[[:print:]]*-", "", dispatch[,1])
  
  dispatch[,"id"] <- NULL
  
  dispatch <- melt(dispatch, id.vars=c("date", "zone", "plant"))
  
  colnames(dispatch) <- c("date", "zone", "plant", "hour", "MW")
  #print(dispatch)
  
  # drop rows with zero generation
  #dispatch <- dispatch[dispatch$MW != 0,]
  # match with fuel type
  dispatch <- merge(dispatch, gens[,c("Name", "Category")], by.x="plant", by.y="Name", all.x=T)
  
  #drop duplicated entries in output
  dispatch <- dispatch[!duplicated(dispatch), ]
  
  # summarize by fuel type
  fuelDispatch <- ddply(dispatch, ~ date + hour + zone + Category, summarise, MW = sum(MW))
  fuelDispatch$zone <- factor(fuelDispatch$zone)
  fuelDispatch$datetime <- as.POSIXct(with(fuelDispatch, paste(date, hour)), format = "%Y-%m-%d %H")
  
  fuelDispatch$Category <- factor(fuelDispatch$Category, levels = c("Sync_Cond","Solar RTPV","Solar PV","CSP","Wind",
                                                                    "Oil ST","Oil CT","Gas CT","Gas CC",
                                                                    "Coal","Hydro","Nuclear"))
  #create color panel for later use
  dispatchcolors <- c("gray50","yellow","yellow","red","cyan",
                      "green","green","brown","orange","grey","blue","purple")
  
  #drop NA's
  fuelDispatch <- fuelDispatch[!is.na(fuelDispatch$Category),]
  
  #fuelDispatch$Category <- mapvalues()
  
  # calculate all zones
  all_dispatch <- ddply(fuelDispatch, ~ datetime + Category, summarise, MW = sum(MW))
  all_dispatch$zone <- "All Zones"
  
  fuelDispatch <- fuelDispatch[,c("datetime", "Category", "MW", "zone")]
  fuelDispatch <- rbind(fuelDispatch, all_dispatch)
  
  #fuelDispatch 
  #fuelDispatch$Category <- droplevels(fuelDispatch$Category)

  #Luke's plotting code (active)
  ggplot(data=fuelDispatch, aes(x=datetime, y=MW, fill=Category)) + geom_area() + 
    geom_line(data = fuelemissions, aes(x=datetime, y = Emissions,linetype=''),colour='black',size=2) +
    facet_wrap(~zone, nrow=3, scales = "free") + 
    theme_classic() + ylab("Gen (MW) or Emissions (tonne)") +
    guides(fill=guide_legend(title="Gen: ", nrow=2, byrow=TRUE),
           linetype=guide_legend(title="Emissions: ")) +
    xlab("") + scale_x_datetime() + scale_fill_manual(values=dispatchcolors) +
    theme(legend.title = element_text(size=32),
          legend.text = element_text(size=28),
          legend.position = "bottom",
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=16),
          axis.text.y= element_text(size=16),
          strip.text.x = element_text(size = 24)) +
    ggtitle(paste("Generation by fuel for", plotTitle))
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("dispatch ", plotTitle, ".png"), width=20, height=12)
  
  return(fuelDispatch)
}
#plotDispatch(results1,dates1,plotTitle='test')
#compare conventional dispatch

compareplotDispatch <- function(dispatch_df1,dispatch_df2){
  dispatchdelta_df <- dispatch_df1[,c("datetime","Category","zone")]
  dispatchdelta_df$MW <- dispatch_df1$MW - dispatch_df2$MW
  
  dispatchdelta_df$Category <- factor(dispatchdelta_df$Category, levels = c("Sync_Cond","Solar RTPV","Solar PV","CSP","Wind",
                                                                    "Oil ST","Oil CT","Gas CT","Gas CC",
                                                                    "Coal","Hydro","Nuclear"))
  #create color panel for later use
  dispatchcolors <- c("gray50","yellow","yellow","red","cyan",
                      "green","green","brown","orange","black","blue","purple")
  
  
  #Luke's plotting code (active)
  ggplot(data=dispatchdelta_df, aes(x=datetime, y=MW, fill=Category)) + geom_area() + facet_wrap(~zone, nrow=3, scales = "free") + 
    theme_classic() + ylab("MW") + guides(fill=guide_legend(title="", nrow=2, byrow=TRUE)) + xlab("") +
    scale_x_datetime() + scale_fill_manual(values=dispatchcolors) +
    theme(legend.text = element_text(size=32),
          legend.position = "bottom",
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=16),
          axis.text.y= element_text(size=16),
          strip.text.x = element_text(size = 24)) +
    ggtitle(paste("Generation by fuel deltas"))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("dispatch delta", ".png"), width=20, height=12)
}


#storage dispatch
plotStorage <- function(results, dates, plotTitle, hours=24){
  storage_dispatch <- results[["storage"]]
  storage_dispatch$datetime <- as.POSIXct(with(storage_dispatch, paste(date, time)), format = "%Y-%m-%d %H")
  
  #Luke's plotting code (active)
  ggplot(data=storage_dispatch, aes(x=datetime, y=soc, fill="SOC")) + geom_area(alpha=0.5) + 
    geom_line(aes(datetime, dispatch,color='Storage Dispatch'),lwd=3) +
    geom_line(aes(datetime, lmp,color='LMP'),lwd=2,linetype='dashed') +
    theme_classic() + ylab("MWh or LMP ($/MWh)") + xlab("") +
    scale_x_datetime() +
    scale_color_manual(name = "", values = c("Storage Dispatch" = "black", "LMP" = "red")) +
    scale_fill_manual(name="",values=c("SOC"="gray50")) +
    guides(color=guide_legend(nrow=1)) +
    theme(legend.text = element_text(size=32),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Storage Dispatch ", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage dispatch ", plotTitle, ".png"), width=20, height=12)
  
  return(storage_dispatch)
}

#compare storage dispatch
compareplotStorage <- function(storage_df1,storage_df2,plotTitle='NA'){
  storagedelta_df <- storage_df1[,c("X","time","date","datetime")]
  storagedelta_df$dispatch <- storage_df1$dispatch - storage_df2$dispatch
  storagedelta_df$soc <- storage_df1$soc - storage_df2$soc
  storagedelta_df$lmp <- storage_df1$lmp - storage_df2$lmp
  
  ggplot(data=storagedelta_df, aes(x=datetime, y=soc, fill="SOC")) + geom_area(alpha=0.5) + 
    geom_line(aes(datetime, dispatch,color='Storage Dispatch'),lwd=3) +
    geom_line(aes(datetime, lmp,color='LMP'),lwd=2,linetype='dashed') +
    theme_classic() + ylab("MWh or LMP ($/MWh)") + xlab("") +
    scale_x_datetime() +
    scale_color_manual(name = "", values = c("Storage Dispatch" = "black", "LMP" = "red")) +
    scale_fill_manual(name="",values=c("SOC"="gray50")) +
    guides(color=guide_legend(nrow=1)) +
    theme(legend.text = element_text(size=32),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Storage Delta Dispatch ", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage delta dispatch",".png"), width=20, height=12)
  
  }

compareStorageHeatplot <- function(storagedflist,plotTitle='NA'){
  #eventually probably a list of dfs as input
  for (i in 1:length(storagedflist)){
    storagedflist[[i]]$label <- names(storagedflist[i]) #may want better label
  }
  storagedf <- do.call("rbind", storagedflist)
  #storagedf1$label <- "Jan 1-7"
  #storagedf2$label <- "Jan 8-14"
  #storagedf <- rbind(storagedf1, storagedf2)
  #print(storagedf)
  
  #geom_tile(aes(fill = value)) + 
  #geom_text(aes(label = round(value, 1))) +
  #+ scale_y_continuous(trans='reverse',breaks=c(2,4,6,8,10,12))
  ggplot(data = storagedf, aes(x = time, y = label, fill = dispatch)) +
    geom_tile() + geom_text(aes(label=round(dispatch,0),size=0.5))+
    theme_classic() +
    scale_fill_gradient2(low = "darkred",mid="white",high = "darkgreen",
                         midpoint=0,na.value = "grey",limits = c(-50,50)) +
    labs(fill="Dispatch \n (MWh)",x="Hour") +
    theme(legend.position="right",
          text = element_text(size=28),
          legend.text=element_text(size=28),
          legend.key.size = unit(3,"line"),
          axis.text.x = element_text(size=24),
          axis.text.y = element_text(size=24),
          axis.title.y = element_text(size=28),
          axis.title.x = element_text(size=28))+
    ggtitle(paste("Storage Dispatch Heatplot ", plotTitle))
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage heatplot ",plotTitle,".png"), width=16, height=6)
}

compareStorageProfit <- function(storagedflist,plotTitle='hi'){
  #eventually probably a list of dfs as input
  print(names(storagedflist))
  for (i in 1:length(storagedflist)){
    storagedflist[[i]]$label <- names(storagedflist[i]) #may want better label
  }
  storagedf <- do.call("rbind", storagedflist)
  
  #sum profit by day for plotting
  storagedf <- ddply(storagedf, ~ date + label, summarise, profit = sum(profit))
  
  print(storagedf)
  #geom_bar(stat='identity',
  ggplot(data=storagedf, aes(x=date, y=profit, fill=label)) +
    geom_bar(stat='identity',position='dodge') + 
    theme_classic() + ylab("Profit($)") + xlab("") +
    theme(legend.text = element_text(size=32),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Storage Profit ", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage profit plot",plotTitle,".png"), width=12, height=6)
  
}

cleanOffer <- function(results,dates,hours=24){
  offer <- results[['offer']]
  gens <- results[['gens']]
  
  offer$segID <- gsub("[[:print:]]*-", "", offer[,1])
  offer$genID <- gsub("-\\d","",offer[,1])
  offer$area <- substr(offer$Zone,start=1,stop=1)
  
  offer <- merge(offer, gens[,c("Name", "Category")], by.x="genID", by.y="Name", all.x=T)
  #drop duplicated entries in output
  offer <- offer[!duplicated(offer), ]
  #print(offer)
  profits <- ddply(offer, ~ date + area, summarise, Profits = sum(Profit))
  return(profits)
}

compareGeneratorProfit <- function(generatordflist,plotTitle='hi'){
  #eventually probably a list of dfs as input
  #print(names(generatordflist))
  for (i in 1:length(generatordflist)){
    generatordflist[[i]]$label <- names(generatordflist[i]) #may want better label
  }
  gendf <- do.call("rbind", generatordflist)
  
  ggplot(data=gendf, aes(x=date, y=Profits, fill=label)) +
    geom_bar(stat='identity',position='dodge') + 
    theme_classic() + ylab("Profit($)") + xlab("") +
    theme(legend.text = element_text(size=32),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Generator Profit ", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("generator profit plot",plotTitle,".png"), width=12, height=6)
}

cleanOffer(results1,dates1) #takes awhile bc big file


dates1 <- seq(as.POSIXct("1/1/2019", format = "%m/%d/%Y"), by="day", length.out=15)
#dates2 <- seq(as.POSIXct("1/3/2019", format = "%m/%d/%Y"), by="day", length.out=1)

results1 <- loadAllCases(dates1,folder='basedispatch')
#results2 <- loadAllCases(dates2)

results1competitive <- loadAllCases(dates1,folder="test")

d1 <- plotDispatch(results1,dates1,plotTitle='Jan 1-15 2019')
d2 <- plotPrices(results1,dates1,plotTitle='Jan 1-15 2019')
d3 <- plotStorage(results1,dates1,plotTitle='Jan 1-15 2019')

c1 <- plotDispatch(results1competitive,dates1,plotTitle='Jan 1-15 2019 competitive')
c2 <- plotPrices(results1competitive,dates1,plotTitle='Jan 1-15 2019 competitive')
c3 <- plotStorage(results1competitive,dates1,plotTitle='Jan 1-15 2019 competitive')

compareplotDispatch(d1,c1)
compareplotPrices(d2,c2)
compareplotStorage(d3,c3)
compareStorageHeatplot(list(d3,c3))

#July
datesJuly <- seq(as.POSIXct("7/1/2019", format = "%m/%d/%Y"), by="day", length.out=30)
results1July <- loadAllCases(datesJuly, folder='basedispatch')
resultscompetitiveJuly <- loadAllCases(datesJuly,folder='test')

j1 <- plotDispatch(results1July,datesJuly,plotTitle='July 1-30 2019')
j2 <- plotPrices(results1July,datesJuly,plotTitle='July 1-30 2019')
j3 <- plotStorage(results1July,datesJuly,plotTitle='July 1-30 2019')

jc1 <- plotDispatch(resultscompetitiveJuly,datesJuly,plotTitle='July 1-30 2019 competitive')
jc2 <- plotPrices(resultscompetitiveJuly,datesJuly,plotTitle='July 1-30 2019 competitive')
jc3 <- plotStorage(resultscompetitiveJuly,datesJuly,plotTitle='July 1-30 2019 competitive')

compareplotDispatch(j1,jc1)
compareplotPrices(j2,jc2)
compareplotStorage(j3,jcc3)
compareStorageHeatplot(list(j3,jc3))

#comparisons on Jan 6
Jan6 <- seq(as.POSIXct("1/6/2019", format = "%m/%d/%Y"), by="day", length.out=1)
base <- loadAllCases(Jan6, folder='basedispatch')
competitive <- loadAllCases(Jan6,folder='test')
storage1 <- loadAllCases(Jan6,folder='storage1')
lessthan35MWh3 <- loadAllCases(Jan6,folder='3lessthan35mwh')
only316 <- loadAllCases(Jan6,folder='316')
only316_318_321 <- loadAllCases(Jan6,folder='316')
all2 <- loadAllCases(Jan6,folder='all2')

#3lessthan35mwh
#316
#316_318_321

basedispatch
competitivedispatch


basestorage <- plotStorage(base,Jan6,plotTitle='Base6Jan')
competitivestorage <- plotStorage(competitive,Jan6,plotTitle='Competitive6Jan')
storage1storage <- plotStorage(storage1,Jan6,plotTitle='Storage16Jan')
lessthan35MWh3storage <- plotStorage(lessthan35MWh3,Jan6,plotTitle='lessthan35mwh3')
only316storage <- plotStorage(only316,Jan6,plotTitle='316')
only316_318_321storage <- plotStorage(only316_318_321,Jan6,plotTitle='316_318_321')
all2storage <- plotStorage(all2,Jan6,plotTitle='all2')

#compareplotStorage(competitivestorage,storage1storage,plotTitle='same?')

caselist <- list(basestorage,competitivestorage,storage1storage,
                 lessthan35MWh3storage,only316storage,only316_318_321storage,
                 all2storage)
names(caselist) <- c('base','competitive','storage1',
                     'lessthan35MWh','only316','316_318,321','all2')
compareStorageProfit(caselist,plotTitle='test')
compareStorageHeatplot(caselist,plotTitle='test')


caselist <- list(cleanOffer(base,Jan6),cleanOffer(competitive,Jan6),cleanOffer(storage1,Jan6),
                 cleanOffer(lessthan35MWh3,Jan6),cleanOffer(only316,Jan6),cleanOffer(only316_318_321,Jan6),
                 cleanOffer(all2,Jan6))
names(caselist) <- c('base','competitive','storage1',
                     'lessthan35MWh','only316','316_318,321',
                     'all2')
compareGeneratorProfit(caselist,plotTitle='test')

#other storage profit comparisons
caselist <- list(j3,jc3)
names(caselist) <- c("Base",'Competitive')
compareStorageProfit(caselist,plotTitle='July')

caselist <- list(d3,c3)
names(caselist) <- c("Base",'Competitive')
compareStorageProfit(caselist,plotTitle='January')

