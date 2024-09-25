# Prepare a color palette. Here with R color brewer:
library(RColorBrewer)
library(ggplot2)
library(scales)
library(ggsci)

###########################################################################
############### Pie chart
###########################################################################

df <- data.frame(value = c(6148, 0), Category = c("Spanish","Synthetic"))
#df <- data.frame(value = c(1150, 5000), Category = c("Spanish","Synthetic"))

#myPalette <- brewer.pal(5, "Set2")
myPalette <- c("#003462","#72CAC3")

p <- ggplot(df, aes(x = "", y = value, fill = Category)) +
  geom_col(color = "grey") +
  #geom_col() +
  geom_text(aes(label = value), color = "white", size = 8, position = position_stack(vjust = 0.5)) + 
  coord_polar(theta = "y") +
  #scale_fill_manual(values = myPalette) +
  scale_fill_manual(values = myPalette) +
  theme_void() +
  theme(text = element_text(size = 17))

p

###########################################################################
############### Bar chart
###########################################################################

df <- read.csv("overall_performance.csv")

subcols <- c("Data", "Model", "F.score")
sub_df <- df[subcols]

models <- unique(sub_df$Model)
datamde <- unique(sub_df$Data)

# Increase bottom margin
par(mar=c(5,4,4,4))

#myPalette <- c("#90d8b2","#8dd2dd","#8babf1","#8b95f6","#9b8bf4")
#myPalette <- c("#f6d094","#eef396","#addd95","#9bd8b2","#8fd4c8")
#myPalette <- c("#f6d094","#eef396","#addd95","#76cb99","#8da0ec")
#myPalette <- c("#2e6eba","#eeaa09","#c20383","#7f5dc3","#e17324")
myPalette <- c("#e54d36","#45a9ba","#3aa088","#3c5586","#f09b81")


p <- ggplot(sub_df, aes(x = Data, y = F.score, fill = Model)) + 
  #scale_x_continuous("", breaks=c(-0.25,0.25,0.75,1.25,1.75,2.25,2.75,3.25), labels=rep(c("Female","Male"), times=4)) +
  scale_fill_manual(values = myPalette) +
  geom_col(aes(y = F.score), position = "dodge") +
  geom_text(aes(label = round(F.score, digits = 1)), 
            vjust = -0.7, 
            size = 4,
            position = position_dodge(width = 0.9)) +
  theme_bw() +
  #theme_update(axis.ticks.x = element_blank(), axis.text.x = element_blank()) +
  ylim(0, 100) +
  theme(plot.margin = unit(c(1,2,3,1), "lines"), 
        text = element_text(size = 18, family="PT Sans"),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        #legend.position="none"
        legend.direction='horizontal'
        )

p





