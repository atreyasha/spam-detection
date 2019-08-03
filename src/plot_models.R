#!/usr/bin/env Rscript
# -*- coding: utf-8 -*- 

# import dependencies
library(optparse)
library(ggplot2)
library(latex2exp)
library(extrafont)
# only required for initial run
# font_install('fontcm')
loadfonts()

###########################
# method for combined plot
###########################

combinedPlot <- function(){
  # make mapping for nice names
  mapping <- c("CNN-LSTM (Words)"="rnn_words_random_embed","CNN-LSTM (Words+Characters)"="rnn_all_random_embed",
               "CNN-LSTM (Words+GloVe)"="rnn_words_glove_embed","CNN-LSTM (Words+Characters+GloVe)"="rnn_all_glove_embed",
               "SVM (Linear Kernel)"="svm_linear","SVM (RBF Kernel Approximation)"="svm_rbf")
  # read in all files and merge into single dataframe
  files <- list.files("./pickles",pattern="precision_recall.*",recursive=TRUE,full.names=TRUE)
  store <- lapply(files, function(x) {
    store <- read.csv(x,stringsAsFactors=FALSE)
    store["name"] <- names(mapping)[which(mapping==gsub("\\/.*","",gsub("(.*)(rnn|svm.*)","\\2",x)))]
    store <- store[c(4,1:3)]
    return(store)
  })
  optimal <- lapply(files, function(x) {
    store <- read.csv(x,stringsAsFactors=FALSE)
    store["name"] <- names(mapping)[which(mapping==gsub("\\/.*","",gsub("(.*)(rnn|svm.*)","\\2",x)))]
    store <- store[c(4,1:3)]
    if(sum(which(store["precision"]>=0.998)) == 0){
      store <- store[which(store["precision"]==max(store["precision"])),]
    } else {
      store <- store[which(store["precision"]>=0.998),]
      store <- store[which(store["recall"]==max(store["recall"])),]
    }
    store$optimal <- "optimal"
    return(store)
  })
  store <- do.call("rbind",store)
  optimal <- do.call("rbind",optimal)
  store <- store[-which(store[,2]==0),]
  store$name <- factor(store$name,levels = names(mapping)[c(1,3,5,2,4,6)])
  optimal$name <- factor(optimal$name,levels = names(mapping)[c(1,3,5,2,4,6)])
  # prepare ggplot for combined object
  pdf("../img/combined.pdf", width=12, height=8)
  g <- ggplot(data=store) + 
    geom_point(aes(x=recall, y=precision, colour=threshold),size=1.5) +
    geom_point(data = optimal, aes(x=recall, y=precision, fill=optimal),size=2, colour="red", alpha = 0.8) +
    geom_abline(slope = 0, intercept = 0.998, linetype = "dashed", size = 0.25, colour = "red") +
    theme_bw() +
    ylim(c(min(store["precision"]),1.00)) + xlim(c(min(store["recall"]),1.00)) +
    xlab("\nRecall") + ylab("Precision\n") + 
    theme(text = element_text(size=13, family="CM Roman"),
          legend.text=element_text(size=10),
          legend.title=element_text(size=10),
          legend.key = element_rect(colour = "lightgray", fill = "white"),
          strip.text = element_text(size = 10)) +
    scale_colour_continuous(type = "viridis", guide = "colourbar") +
    scale_fill_discrete(name="",
                        breaks=c("optimal"),
                        labels=c("Optimal\nThreshold")) +
    facet_wrap(~name, ncol=3) +
    guides(colour = guide_colourbar(title="Threshold", barwidth = 0.9, barheight = 18))
  # print object
  print(g)
  dev.off()
  embed_fonts("../img/combined.pdf", outfile="../img/combined.pdf")
}

###########################
# main command call
###########################

combinedPlot()