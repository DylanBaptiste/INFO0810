library(plotly)

dataset = data.frame(read.csv("./data/import_export/normal_reel.csv", header = TRUE))
plot_ly(dataset, x=1:nrow(dataset)) %>%
	add_trace(y=~conv_debut, name="fermee", type='scatter', mode='lines')


data = data.frame(read.csv("./resultat/import_export/history.csv", header = TRUE))
data = data.frame(read.csv("./resultat/import_export/history2.csv", header = TRUE))
colnames(data)
p <- plot_ly(data, x=1:nrow(data)) %>% add_segments(x = 0, xend = nrow(data), y = 1, yend = 1)
for(el in colnames(data)){
	p <- p %>% add_trace(y=data[[el]], name=el, type='scatter', mode='lines')
}
p <- p %>% layout(yaxis = list(range = c(0, 1)))
p


########################################
element = "fermee"
plot_ly(data, x=1:nrow(data)) %>%
	add_trace(y=data[[element]], name=element, type='scatter', mode='lines') %>%
	add_trace(y=data[[paste("val_", element, sep="")]], name=paste("val_", element, sep=""), type='scatter', mode='lines')


