rm(list=ls())

require(ggplot2)
require(openxlsx)
require(plyr)
require(lubridate)
library(reshape2)
library(skimr)
library(dplyr)
#library(sjPlot)
library(table1)
library(xtable)

baseWD <- "C:/Users/wenmi/Desktop/competitiveMPEC"
setwd(paste(baseWD, "", sep="/"))

## Load model results ####

# helper function to read all files in date range of a specific output
readFiles <- function(filename, dates, dateWD, subFolder="results_DA"){
  
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

readFilesRT <- function(filename, dates, dateWD, tmps, slicer){
  for(i in 1:length(dates)){
    for(j in 1:slicer){
      date <- dates[i]
      subFolder <- paste("results_RT", tmps * (j - 1) + 1, tmps * j, sep="_")
      dateString <- paste(as.character(format(date, "%m")), as.character(format(date, "%d")), as.numeric(format(date, "%Y")), sep=".")
      setwd(paste(dateWD, dateString, subFolder, sep="/"))
    
      # load file
      dftmp <- read.csv(filename)
      dftmp$date <- date
      if(j == 1){
        df <- dftmp
      } else{
        df <- rbind(df, dftmp)
      }
    }
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
  dateResultsWD<-paste(baseWD, folder, sep="/")
  #if (length(dateMonth) == 1){
  #  dayStart <- as.numeric(format(dates[1], "%d"))
  #  dayEnd <- max(as.numeric(format(dates[length(dates)], "%d")),dayStart+6)
    #dayEnd <- dayEnd
    # case options: NoReserves, SimpleORDC, ConstantReserves, withORDC
    # there will be new case options but this should be ok
    #directory <- paste(dateMonth, dayStart, dayEnd, dateYear, case, sep="_")
  
    
  #} else{
  #  print("Error. Multiple months in date range.")   # multiple months yields error for now, can update based on directory format
  #  return(NULL)
  #}
  
  modelLMP <- readFiles("zonal_prices.csv", dates, dateResultsWD)
  txFlows <- readFiles("tx_flows.csv", dates, dateResultsWD)
  offer <- readFiles("generator_segment_offer.csv", dates, dateResultsWD)
  dispatch <- readFiles("generator_dispatch.csv", dates, dateResultsWD)
  storage <- readFiles("storage_dispatch.csv", dates, dateResultsWD)
  #VRE <- readFiles("renewable_generation.csv", dates, dateResultsWD)
  
  #ordc <- readFiles("full_ordc.csv", dates, dateResultsWD, subFolder="inputs")
  gens <- readFiles("generators_descriptive.csv", dates, dateResultsWD, subFolder="inputs")
  zonalLoad <- readFiles("timepoints_zonal.csv", dates, dateResultsWD, subFolder="inputs")
  #emissions <- readFiles("generator_segment_marginalcost.csv", dates, dateResultsWD, subFolder="inputs")

  # some formatting
  #gens <- gens[!duplicated(gens),] # remove duplicate generators
  #gens <- gens[,c("Name", "Zone", "Category")]  # subset generator columns
  
  #reformat zonal prices so it matches previous format
  #modelLMP <- modelLMP[,c("X","hour","LMP","date")]
  
  # return results
  results <- list(modelLMP, zonalLoad, dispatch, gens, txFlows, storage, offer)
  names(results) <- c("modelLMP", "zonalLoad", "dispatch", "gens", "txFlows", "storage", "offer")
  return(results)
}

loadResultsRT <- function(dates,folder){
  dateMonth <- unique(format(dates, "%b"))  # extract month of dates for directory
  dateYear <- unique(format(dates, "%Y"))  # extract year of dates for directory
  dateResultsWD<-paste(baseWD, folder, sep="/")

  gens <- readFiles("generators_descriptive.csv", dates, dateResultsWD, subFolder="inputs")
  zonalLoad <- readFiles("timepoints_zonal.csv", dates, dateResultsWD, subFolder="inputs")
  modelLMPRT <- readFilesRT("zonal_prices.csv", dates, dateResultsWD, tmps = 48, slicer = 6)
  txFlowsRT <- readFilesRT("tx_flows.csv", dates, dateResultsWD, tmps = 48, slicer = 6)
  offerRT <- readFilesRT("generator_segment_offer.csv", dates, dateResultsWD, tmps = 48, slicer = 6)
  dispatchRT <- readFilesRT("generator_dispatch.csv", dates, dateResultsWD, tmps = 48, slicer = 6)
  storageRT <- readFilesRT("storage_dispatch.csv", dates, dateResultsWD, tmps = 48, slicer = 6)

  # return resultsRT
  resultsRT <- list(modelLMPRT, zonalLoad, dispatchRT, gens, txFlowsRT, storageRT, offerRT)
  names(resultsRT) <- c("modelLMP", "zonalLoad", "dispatch", "gens", "txFlows", "storage", "offer")
  return(resultsRT)
}

# helper function to load data from all four cases
loadAllCases <- function(dates,folder="test"){
  results <- loadResults(dates,folder)
  return(results,resultsRT)
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

plotPricesRT <- function(results,dates,plotTitle,hours=24,slice=12){
  prices <- results[['modelLMP']]
  prices$zone <- substr(prices[,1],start=1,stop=1)
  prices$zone <- paste0("Area ",prices$zone)
  prices$busID <- substr(prices[,1],start=2,stop=3)
  prices$time <- paste(prices$hour%/%slice,prices$hour%%slice*5,sep=":")
  prices$datetime <- as.POSIXct(with(prices, paste(date, time)), format = "%Y-%m-%d %H:%M")
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
  offer$area <- substr(offer$Zone,start=1,stop=3)

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
  dispatch$zone <- substr(dispatch[,1],start=1,stop=3)
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
          axis.text.x= element_text(size=8),
          axis.text.y= element_text(size=16),
          strip.text.x = element_text(size = 24)) +
    ggtitle(paste("Generation by fuel for", plotTitle))
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("dispatch ", plotTitle, ".png"), width=40, height=12)
  
  return(list(fuelDispatch,fuelemissions))
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

compareTotalEmissions <- function(emissiondflist,plotTitle='hi',resolution=NA){
    #eventually probably a list of dfs as input
    print(names(emissiondflist))
    for (i in 1:length(emissiondflist)){
      emissiondflist[[i]]$label <- names(emissiondflist[i]) #may want better label
    }
    emissionsdf <- do.call("rbind", emissiondflist)
    
    #print(emissionsdf)
    if (resolution=='month'){
      emissionsdf$date <- month(emissionsdf$date)
    }
    #emissionsdf <- emissionsdf[emissionsdf$zone!='All Zones',]
    #emissionsdf$date <- as.Date(emissionsdf$datetime)
    #sum profit by day for plotting
    emissionsdf <- ddply(emissionsdf, ~ date + label, summarise, emissions = sum(Emissions))
    
    #print(storagedf)
    #geom_bar(stat='identity',
    ggplot(data=emissionsdf, aes(x=date, y=emissions, fill=label)) +
      geom_bar(stat='identity',position='dodge') + 
      theme_classic() + ylab("Emissions (tonne CO2)") + xlab("") +
      guides(fill=guide_legend(title="")) +
      theme(legend.text = element_text(size=32),
            legend.position = 'bottom',
            plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
            axis.title.y = element_text(size=28),
            axis.text.x= element_text(size=20),
            axis.text.y= element_text(size=20)) +
      ggtitle(paste("Total Emissions Comparison"))
    
    setwd(paste(baseWD, "post_processing", "figures", sep="/"))
    ggsave(paste0("total emissions plot",plotTitle,".png"), width=12, height=6)
    return(emissionsdf)
}

compareTotalGeneratorCost <- function(generatordflist,plotTitle='hi',resolution='NA'){
  #eventually probably a list of dfs as input
  print(names(generatordflist))
  for (i in 1:length(generatordflist)){
    generatordflist[[i]]$label <- names(generatordflist[i]) #may want better label
  }
  generatordf <- do.call("rbind", generatordflist)
  
  if (resolution=='month'){
    generatordf$date <- month(generatordf$date)
  }
  #print(generatordf)

  generatordf <- ddply(generatordf, ~ date + label, summarise, cost = sum(Cost))
  
  #print(storagedf)
  #geom_bar(stat='identity',
  #generatordf$cost <- as.numeric(generatordf$cost)
  #    scale_y_continuous(limits = c(30000, 40000),oob = rescale_none) +
  ggplot(data=generatordf, aes(x=date, y=cost, fill=label)) +
    geom_bar(stat='identity',position='dodge') + 
    theme_classic() + ylab("Generator Revenue ($)") + xlab("") +
    guides(fill=guide_legend(title="")) +
    theme(legend.text = element_text(size=32),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=28),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Wind Bus 309 Revenue Comparison"))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("total dispatch cost plot",plotTitle,".png"), width=12, height=6)
  return(generatordf)
}


#storage dispatch
plotStorageRT <- function(results, dates, plotTitle, hours=24,slice=12){
  storage_dispatch <- results[["storage"]]
  storage_dispatch$time <- paste(storage_dispatch$time%/%slice,storage_dispatch$time%%slice*5,sep=":")
  storage_dispatch$datetime <- as.POSIXct(with(storage_dispatch, paste(date, time)), format = "%Y-%m-%d %H:%M")
  storage_dispatch$dispatch <- storage_dispatch$discharge-storage_dispatch$charge
  storage_dispatch$X <- factor(storage_dispatch$X)
  storage_dispatch$node <- factor(storage_dispatch$node)
  
  #Luke's plotting code (active)
  ggplot(data=storage_dispatch, aes(x=datetime, y=soc, fill=X)) + geom_area(alpha=0.5) + 
    geom_line(aes(datetime, dispatch, color=X),lwd=3) +
    geom_line(aes(datetime, lmp, color=node),lwd=2,linetype='dashed') +
    theme_classic() + ylab("MWh or LMP ($/MWh)") + xlab("") +
    scale_x_datetime() +
    #scale_color_grey() +     define line colors here
    scale_fill_grey() +
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

#storage dispatch
plotStorage <- function(results, dates, plotTitle, hours=24){
  storage_dispatch <- results[["storage"]]
  storage_dispatch$datetime <- as.POSIXct(with(storage_dispatch, paste(date, time)), format = "%Y-%m-%d %H")
  storage_dispatch$dispatch <- storage_dispatch$discharge-storage_dispatch$charge
  storage_dispatch$X <- factor(storage_dispatch$X)
  storage_dispatch$node <- factor(storage_dispatch$node)
  
  #Luke's plotting code (active)
  ggplot(data=storage_dispatch, aes(x=datetime, y=soc, fill=X)) + geom_area(alpha=0.5) + 
    geom_line(aes(datetime, dispatch, color=X),lwd=3) +
    geom_line(aes(datetime, lmp, color=node),lwd=2,linetype='dashed') +
    theme_classic() + ylab("MWh or LMP ($/MWh)") + xlab("") +
    scale_x_datetime() +
    #scale_color_grey() +     define line colors here
    scale_fill_grey() +
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

compareStorageHeatplot <- function(storagedflist,plotTitle='NA',type='NA'){
  #eventually probably a list of dfs as input
  for (i in 1:length(storagedflist)){
    storagedflist[[i]]$label <- names(storagedflist[i]) #may want better label
  }
  storagedf <- do.call("rbind", storagedflist)
  #print(storagedf)
  #+ scale_y_continuous(trans='reverse',breaks=c(2,4,6,8,10,12))
  #,limits = c(-50,50
  if (type=='lmp'){
    ggplot(data = storagedf, aes(x = time, y = label, fill = lmp)) +
      geom_tile() + geom_text(aes(label=round(lmp,0)))+
      theme_classic() +
      scale_fill_gradient2(low = "darkgreen",mid='yellow',high = "darkred",
                           na.value = "grey") +
      labs(fill="LMP ($/MWh)",x="Hour") +
      theme(legend.position="right",
            text = element_text(size=14),
            legend.text=element_text(size=28),
            legend.key.size = unit(3,"line"),
            axis.text.x = element_text(size=24),
            axis.text.y = element_text(size=24),
            axis.title.y = element_text(size=28),
            axis.title.x = element_text(size=28))+
      ggtitle(paste("Storage LMP Heatplot ", plotTitle))
  }
  else{
    ggplot(data = storagedf, aes(x = time, y = label, fill = dispatch)) +
      geom_tile() + geom_text(aes(label=round(dispatch,-1)))+
      theme_classic() +
      scale_fill_gradient2(low = "darkred",mid="white",high = "darkgreen",
                           midpoint=0,na.value = "grey") +
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
  }
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage heatplot ",plotTitle,".png"), width=16, height=6)
}

compareStorageProfit <- function(storagedflist,plotTitle='hi',resolution=NA){
  #eventually probably a list of dfs as input
  print(names(storagedflist))
  for (i in 1:length(storagedflist)){
    storagedflist[[i]]$label <- names(storagedflist[i]) #may want better label
  }
  storagedf <- do.call("rbind", storagedflist)
  
  if (resolution=='month'){
    storagedf$date <- month(storagedf$date)
  }
  #sum profit by day for plotting
  storagedf <- ddply(storagedf, ~ date + label, summarise, profit = sum(profit))
  
  #print(storagedf)
  #geom_bar(stat='identity',
  ggplot(data=storagedf, aes(x=date, y=profit, fill=label)) +
    geom_bar(stat='identity',position='dodge') + 
    theme_classic() + ylab("Profit($)") + xlab("") +
    theme(legend.text = element_text(size=26),
          legend.position = 'bottom',
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          axis.title.y = element_text(size=32),
          axis.text.x= element_text(size=20),
          axis.text.y= element_text(size=20)) +
    ggtitle(paste("Storage Profit ", plotTitle))
  
  setwd(paste(baseWD, "post_processing", "figures", sep="/"))
  ggsave(paste0("storage profit plot",plotTitle,".png"), width=12, height=6)
  return(storagedf)
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

#cleanOffer(results1,dates1) #takes awhile bc big file

cleanEmissions <- function(results,dates,hours=24){
  offer <- results[['offer']]
  #gens <- results[['gens']]
  
  offer$segID <- gsub("[[:print:]]*-", "", offer[,1])
  offer$genID <- gsub("-\\d","",offer[,1])
  offer$area <- substr(offer$Zone,start=1,stop=1)

  emissions <- ddply(offer,~date,summarise,Emissions=sum(SegmentEmissions))
  return(emissions)
}

cleanDispatchCost <- function(results,dates,type='NA',filter='None',hour=24){
  offer <- results[['offer']]
  #print(offer)
  offer$segID <- gsub("[[:print:]]*-", "", offer[,1])
  offer$genID <- gsub("-\\d","",offer[,1])
  offer$area <- substr(offer$Zone,start=1,stop=1)
  if (type=='lmp'){
    offer$dispatchcost <- offer$SegmentDispatch*offer$LMP
  }
  else{
    offer$dispatchcost <- offer$SegmentDispatch*offer$MarginalCost
  }
  #print(offer)
  if (filter!='None'){
    offer <- offer[offer$genID==filter,]
  }
  DispatchCost <- ddply(offer,~date,summarise,Cost=sum(dispatchcost))
  return(DispatchCost)
}

dates1 <- seq(as.POSIXct("1/1/2019", format = "%m/%d/%Y"), by="day", length.out=1) # Configure cases period here
#dates2 <- seq(as.POSIXct("1/3/2019", format = "%m/%d/%Y"), by="day", length.out=1)


results1 <- loadResults(dates1,folder='test')
results1RT  <- loadResultsRT(dates1,folder='test')
#results1competitive <- loadAllCases(dates1,folder="competitive")

#results1CO2 <- loadAllCases(dates1,folder='baseCO230')
#results1CO2competitive <- loadAllCases(dates1,folder='competitiveCO230')

d1 <- plotDispatch(results1,dates1,plotTitle='Jan 1 2019')
d2 <- plotPrices(results1,dates1,plotTitle='Jan 1 2019')
d3 <- plotStorage(results1,dates1,plotTitle='Jan 1 2019')

d1RT <- plotDispatch(results1RT,dates1,plotTitle='Jan 1 2019 RT')
d2RT <- plotPricesRT(results1RT,dates1,plotTitle='Jan 1 2019 RT')
d3RT <- plotStorageRT(results1RT,dates1,plotTitle='Jan 1 2019 RT')

c1 <- plotDispatch(results1competitive,dates1,plotTitle='Jan 1-30 2019 competitive')
c2 <- plotPrices(results1competitive,dates1,plotTitle='Jan 1-15 2019 competitive')
c3 <- plotStorage(results1competitive,dates1,plotTitle='Jan 1-15 2019 competitive')

CO21 <- plotDispatch(results1CO2,dates1,plotTitle='Jan 1-30 2019 CO2')
CO21competitive <- plotDispatch(results1CO2competitive,dates1,plotTitle='Jan 1-30 2019 CO2')

#fuelemissions$datetime <- as.POSIXct(with(fuelemissions, paste(date, hour)), format = "%Y-%m-%d %H")
compareplotDispatch(d1,c1)
compareplotPrices(d2,c2)
compareplotStorage(d3,c3)
compareStorageHeatplot(list(d3,c3))


caselist <- list(d1[[2]],c1[[2]],CO21[[2]],CO21competitive[[2]])
names(caselist) <- c('base','competitive','baseCO2','competitiveCO2')
df <- compareTotalEmissions(caselist,plotTitle='test')
write.csv(df,'check.csv')

#all year
dyear <- plotDispatch(results1,dates1,plotTitle='All Year 2019')

dyearprice <- plotPrices(results1,dates1,plotTitle = 'All Year 2019')
dyearpricecompetitive <- plotPrices(results1competitive,dates1,plotTitle='All Year Competitive 2019')
dyearpriceCO2 <- plotPRices(results)

dyearstorage <- plotStorage(results1,dates1,plotTitle='All Year Storage 2019')
dyearstoragecompetitive <- plotStorage(results1competitive,dates1,plotTitle='All Year competitive Storage 2019')
dyearstorageCO2 <- plotStorage(results1CO2, dates1, plotTitle='All Year Storage 2019')

caselist <- list(dyearstorage,dyearstoragecompetitive)
names(caselist) <- c('base','competitive')
compareStorageProfit(caselist,plotTitle='test',resolution='month')
compareStorageHeatplot(caselist,plotTitle='test')
compareStorageHeatplot(caselist,plotTitle='lmp',type='lmp')

caselist <- list(cleanEmissions(results1,dates1),cleanEmissions(results1competitive,dates1),
                 cleanEmissions(results1CO2,dates1))
names(caselist) <- c('base','competitive','CO2')
compareTotalEmissions(caselist,plotTitle='test',resolution='month')

caselist <- list(cleanDispatchCost(results1,dates1),cleanDispatchCost(results1competitive,dates1))
names(caselist) <- c('base','competitive')
compareTotalGeneratorCost(caselist,plotTitle='test',resolution='month')

caselist <- list(cleanDispatchCost(results1,dates1,type='lmp'),
                 cleanDispatchCost(results1competitive,dates1,type='lmp'))
names(caselist) <- c('base','competitive')
compareTotalGeneratorCost(caselist,plotTitle='Payment',resolution='month')

## January only ##
datesJan <- seq(as.POSIXct("1/3/2019", format = "%m/%d/%Y"), by="day", length.out=2)#31

nostorageJan <- loadAllCases(datesJan,folder='NoStorage')
baseJan <- loadAllCases(datesJan,folder='base')
competitiveJan <- loadAllCases(datesJan,folder="competitive")
baseCO2Jan <- loadAllCases(datesJan,folder='baseCO230')
competitiveCO2Jan <- loadAllCases(datesJan,folder='competitiveCO230')
competitiveCCJan <- loadAllCases(datesJan,folder='competitive300CCs')
competitiveSteamJan <- loadAllCases(datesJan,folder='competitive300STEAM')

base303Jan <- loadAllCases(datesJan,folder='BaseStorageBus303')
base309Jan <- loadAllCases(datesJan,folder='BaseStorageBus309')
competitive303Jan <- loadAllCases(datesJan,folder='CompetitiveStorageBus303')
competitive309Jan <- loadAllCases(datesJan,folder='CompetitiveStorageBus309')
competitive303andWind309Jan <- loadAllCases(datesJan,folder='CompetitiveStorageBus303andWind309')

#storage-related plots
#nostorageJan,baseJan,competitiveJan,baseCO2Jan,
#competitiveCO2Jan,competitiveCCJan,competitiveSteamJan

caselist <- list(plotStorage(nostorageJan,datesJan,plotTitle='1'),
                 plotStorage(baseJan,datesJan,plotTitle='2'),
                 plotStorage(competitiveJan,datesJan,plotTitle='3'),
                 plotStorage(base303Jan,datesJan,plotTitle='8'),
                 plotStorage(base309Jan,datesJan,plotTitle='9'),
                 plotStorage(competitive303Jan,datesJan,plotTitle='10'),
                 plotStorage(competitive309Jan,datesJan,plotTitle='11'),
                 plotStorage(competitive303andWind309Jan,datesJan,plotTitle='12'))
names(caselist) <- c('NoStorage','Base313','Comp313',
                     'Base303','Base309','Comp303','Comp309',
                     'Comp303Wind309')

compareStorageProfit(caselist,plotTitle='Jan 3-4th',resolution='na')#resolution='month'
compareStorageHeatplot(caselist,plotTitle='Jan 4th')
compareStorageHeatplot(caselist,plotTitle='Jan',type='lmp')

#lmp-related plots

#generator cost, revenue, and profit
#cost
myfilter ='None'#316_STEAM_1#309_WIND_1
caselist <- list(cleanDispatchCost(nostorageJan,datesJan,filter=myfilter),
                 cleanDispatchCost(baseJan,datesJan,filter=myfilter),
                 cleanDispatchCost(competitiveJan,datesJan,filter=myfilter),
                 cleanDispatchCost(base303Jan,datesJan,filter=myfilter),
                 cleanDispatchCost(base309Jan,datesJan,filter=myfilter),
                 cleanDispatchCost(competitive303Jan,datesJan,filter=myfilter),
                 cleanDispatchCost(competitive309Jan,datesJan,filter=myfilter),
                 cleanDispatchCost(competitive303andWind309Jan,datesJan,filter=myfilter))
names(caselist) <- c('NoStorage','Base','Competitive','Base303','Base309',
                     'Competitive303','Competitive309','Competitive303andWind309')
compareTotalGeneratorCost(caselist,plotTitle='JanCost',resolution='na')#resolution='month'

#payments
#cleanDispatchCost(baseCO2Jan,datesJan,type='lmp')#filter='316_STEAM_1'
myfilter='309_WIND_1'
caselist <- list(cleanDispatchCost(nostorageJan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(baseJan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitiveJan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(base303Jan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(base309Jan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive303Jan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive309Jan,datesJan,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive303andWind309Jan,datesJan,type='lmp',filter=myfilter))
names(caselist) <- c('NoStorage','Base313','Comp313','Base303','Base309',
                     'Comp303','Comp309','Comp303Wind309')
compareTotalGeneratorCost(caselist,plotTitle='JanPmt',resolution='m')#resoultion='month'
#require(scales)
#emissions
caselist <- list(cleanEmissions(nostorageJan,datesJan),
                 cleanEmissions(baseJan,datesJan),
                 cleanEmissions(competitiveJan,datesJan),
                 cleanEmissions(base303Jan,datesJan),
                 cleanEmissions(base309Jan,datesJan),
                 cleanEmissions(competitive303Jan,datesJan),
                 cleanEmissions(competitive309Jan,datesJan),
                 cleanEmissions(competitive303andWind309Jan,datesJan))
names(caselist) <- c('NoStorage','Base313','Comp313','Base303','Base309',
                     'Comp303','Comp309','Comp303Wind309')
compareTotalEmissions(caselist,plotTitle='Jan',resolution='m')

## June only ##
datesJune <- seq(as.POSIXct("6/1/2019", format = "%m/%d/%Y"), by="day", length.out=5)

nostorageJune <- loadAllCases(datesJune,folder='NoStorage')
baseJune <- loadAllCases(datesJune,folder='base')
competitiveJune <- loadAllCases(datesJune,folder="competitive")
baseCO2June <- loadAllCases(datesJune,folder='baseCO230')
competitiveCO2June <- loadAllCases(datesJune,folder='competitiveCO230')
#competitiveCCJune <- loadAllCases(datesJune,folder='competitive300CCs')
competitiveSteamJune <- loadAllCases(datesJune,folder='competitive300STEAM')
base303June <- loadAllCases(datesJune,folder='BaseStorageBus303')
base309June <- loadAllCases(datesJune,folder='BaseStorageBus309')

competitive303June <- loadAllCases(datesJune,folder='CompetitiveStorageBus303')
competitive309June <- loadAllCases(datesJune,folder='CompetitiveStorageBus309')
competitive303andWind309June <- loadAllCases(datesJune,folder='CompetitiveStorageBus303andWind309')
competitive309andWind303June <- loadAllCases(datesJune,folder='CompetitiveStorageBus309andWind303')
#storage-related

caselist <- list(plotStorage(nostorageJune,datesJune,plotTitle='1'),
                 plotStorage(baseJune,datesJune,plotTitle='2'),
                 plotStorage(competitiveJune,datesJune,plotTitle='3'),
                 plotStorage(competitiveSteamJune,datesJune,plotTitle='7'),
                 plotStorage(base303June,datesJune,plotTitle='8'),
                 plotStorage(base309June,datesJune,plotTitle='9'),
                 plotStorage(competitive303June,datesJune,plotTitle='10'),
                 plotStorage(competitive309June,datesJune,plotTitle='11'),
                 plotStorage(competitive303andWind309June,datesJune,plotTitle='12'),
                 plotStorage(competitive309andWind303June,datesJune,plotTitle='13'))
names(caselist) <- c('NoStorage','Base313','Comp313',
                     'CompSteam','Base303','Base309','Comp303','Comp309','Comp303Wind309')

compareStorageProfit(caselist,plotTitle='June',resolution='m')

compareStorageHeatplot(caselist,plotTitle='June')
compareStorageHeatplot(caselist,plotTitle='June',type='lmp')

#generator cost, revenue, and profit
#cost
myfilter='None'
caselist <- list(cleanDispatchCost(nostorageJune,datesJune,filter=myfilter),
                 cleanDispatchCost(baseJune,datesJune,filter=myfilter),
                 cleanDispatchCost(competitiveJune,datesJune,filter=myfilter),
                 cleanDispatchCost(baseCO2June,datesJune,filter=myfilter),
                 cleanDispatchCost(competitiveCO2June,datesJune,filter=myfilter),
                 cleanDispatchCost(competitiveSteamJune,datesJune,filter=myfilter))
names(caselist) <- c('NoStorage','Base313','Comp313',
                     'CompSteam','Base303','Base309',
                     'Comp303','Comp309','Comp303Wind309')
compareTotalGeneratorCost(caselist,plotTitle='JuneCost')#resoultion='month'

#payments
#cleanDispatchCost(baseCO2Jan,datesJan,type='lmp')#filter='316_STEAM_1'
myfilter = '309_WIND_1'
caselist <- list(cleanDispatchCost(nostorageJune,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(baseJune,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitiveJune,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitiveSteamJune,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(base303June,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(base309June,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive303June,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive309June,datesJune,type='lmp',filter=myfilter),
                 cleanDispatchCost(competitive303andWind309June,datesJune,type='lmp',filter=myfilter))
names(caselist) <- c('NoStorage','Base313','Comp313',
                     'CompSteam','Base303','Base309',
                     'Comp303','Comp309','Comp303Wind309')
compareTotalGeneratorCost(caselist,plotTitle='JunePmt_309Wind',resolution='m')#resoultion='month'

#emissions
caselist <- list(cleanEmissions(nostorageJune,datesJune),
                 cleanEmissions(baseJune,datesJune),
                 cleanEmissions(competitiveJune,datesJune),
                 cleanEmissions(baseCO2June,datesJune),
                 cleanEmissions(competitiveCO2June,datesJune),
                 cleanEmissions(competitiveSteamJune,datesJune))
names(caselist) <- c('NoStorage','Base','Competitive','BaseCO2','CompetitiveCO2',
                     'CompetitiveSteam')
compareTotalEmissions(caselist,plotTitle='June',resolution='month')

plotPrices(baseJan,datesJan,plotTitle='check')
