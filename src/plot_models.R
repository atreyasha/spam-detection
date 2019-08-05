#!/usr/bin/env Rscript
# -*- coding: utf-8 -*- 

# import dependencies
library(optparse)
library(ggplot2)
library(latex2exp)
library(extrafont)
library(optparse)
font_install('fontcm')
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
    store <- store[which(store["recall"]!=1),]
    store <- store[which(store[,2]!=0),]
    return(store)
  })
  optimal <- lapply(files, function(x) {
    store <- read.csv(x,stringsAsFactors=FALSE)
    store["name"] <- names(mapping)[which(mapping==gsub("\\/.*","",gsub("(.*)(rnn|svm.*)","\\2",x)))]
    store <- store[c(4,1:3)]
    store <- store[which(store["recall"]!=1),]
    store <- store[which(store["recall"]!=0),]
    if(sum(which(store["recall"]>=0.998)) == 0){
      store <- store[which(store["recall"]==max(store["recall"])),]
    } else {
      store <- store[which(store["recall"]>=0.998),]
      store <- store[which(store["precision"]==max(store["precision"])),]
    }
    store$optimal <- "optimal"
    return(store)
  })
  store <- do.call("rbind",store)
  optimal <- do.call("rbind",optimal)
  store$name <- factor(store$name,levels = names(mapping)[c(1,3,5,2,4,6)])
  optimal$name <- factor(optimal$name,levels = names(mapping)[c(1,3,5,2,4,6)])
  # prepare ggplot for combined object
  pdf("../img/combined.pdf", width=12, height=7)
  g <- ggplot(data=store) + 
    geom_point(aes(x=recall, y=precision, colour=threshold),size=1.5) +
    geom_point(data = optimal, aes(x=recall, y=precision, fill=optimal),size=2, colour="red", alpha = 0.8) +
    geom_vline(xintercept = 0.998, linetype = "dashed", size = 0.25, colour = "red",alpha=0.6) +
    theme_bw() +
    ylim(c(min(store["precision"]),1.00)) + xlim(c(min(store["recall"]),1.00)) +
    xlab("\nRecall") + ylab("Precision\n") + 
    theme(text = element_text(size=13, family="CM Roman"),
          legend.text=element_text(size=10),
          legend.title=element_text(size=10),
          legend.key = element_rect(colour = "lightgray", fill = "white"),
          strip.text = element_text(size = 10,face="bold")) +
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

svmWordPlot <- function(){
  files <- list.files("./pickles",recursive=TRUE,full.names=TRUE)
  files <- grep("ham|spam",grep("linear",files,value=TRUE),value=TRUE)
  dir <- dirname(files[1])
  ham_words <- read.csv(grep("spam.*csv$",files,value=TRUE),stringsAsFactors=FALSE)
  spam_words <- read.csv(grep("ham.*csv$",files,value=TRUE),stringsAsFactors=FALSE)
  ham_words$name <- "bottom"
  spam_words$name <- "top"
  # make plot for top words
  pdf(paste0(dir,"/spam_words.pdf"), width=12, height=8)
  g <- ggplot(data=spam_words, aes(reorder(word,coefficient,sum),coefficient)) + geom_col(fill = "red",colour="black",
                                                                                         size=0.34,alpha = 0.8) +
    theme_bw() + theme(text=element_text(size=16,family="CM Roman"),legend.position="none", plot.title=element_text(hjust = 0.5),
                       axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
                       axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) + 
    xlab("Word Feature") + ylab(TeX("Absolute weight |$\\theta$|")) +
    ggtitle(TeX("Ranked spam |$\\theta$| coefficients"))
  print(g)
  dev.off()
  embed_fonts(paste0(dir,"/spam_words.pdf"), outfile=paste0(dir,"/spam_words.pdf"))
  # make plot for bottom words
  pdf(paste0(dir,"/ham_words.pdf"), width=12, height=8)
  g <- ggplot(data=ham_words, aes(reorder(word,coefficient,sum),coefficient)) + geom_col(fill = "blue",colour="black",
                                                                                         size=0.34,alpha = 0.8) +
    theme_bw() + theme(text=element_text(size=16,family="CM Roman"),legend.position="none", plot.title=element_text(hjust = 0.5),
                       axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
                       axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) + 
    xlab("Word Feature") + ylab(TeX("Absolute weight |$\\theta$|")) +
    ggtitle(TeX("Ranked ham |$\\theta$| coefficients"))
  print(g)
  dev.off()
  embed_fonts(paste0(dir,"/ham_words.pdf"), outfile=paste0(dir,"/ham_words.pdf"))
}

###########################
# main command call
###########################

option_list = list(
  make_option(c("-t", "--type"), type="character", default="combined", 
              help="which process to execute, either 'combined' or 'svm'", metavar="character"))
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
if(opt$type == "combined"){
  combinedPlot()
} else if(opt$type == "svm"){
  svmWordPlot()
}