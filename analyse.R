
data = read.csv("./data/import_export/reframed_3.csv", header = FALSE)
n_features = 34
n_past = 50
n_future = 1

y = data[, ((ncol(data)-n_features*n_future+1):(ncol(data)))]
y = y[, 2:ncol(y)]

library(plotly)
plot_ly(y[1], y=table(y[1]), type="bar")

freq <- t(data.frame(apply(y, 2, table)))[1:(ncol(y)/2)%%2==0,]
freq <- data.frame("0" = freq[,1], "1"=freq[,2])
freq[,] <- lapply(freq[,], function(x) as.numeric(as.character(x)))
freq[18,]$X1 = 0
rownames(freq) <- c(1:33)
freq$freq0 <- freq$X0/(freq$X0+freq$X1)
plot_ly(freq, y=~freq0, type="bar")

data = read.csv("./data/import_export/test", header = FALSE)
n_features = 34
n_past = 50
n_future = 1

y = data[, ((ncol(data)-n_features*n_future+1):(ncol(data)))]
y = y[, 2:ncol(y)]

library(plotly)

plot_ly(y[1], y=table(y[1]), type="bar")

freq <- t(data.frame(apply(y, 2, table)))[1:(ncol(y)/2)%%2==0,]
freq <- data.frame("0" = freq[,1], "1"=freq[,2])
freq[,] <- lapply(freq[,], function(x) as.numeric(as.character(x)))
freq[18,]$X1 = 0
rownames(freq) <- c(1:33)
freq$freq0 <- freq$X0/(freq$X0+freq$X1)
plot_ly(freq, y=~freq0, type="bar")




library(fpp2)
rollmean(data$ZMONT, k=10)
moyenne_glissee <- function(x, n){ return(apply(embed(x, n), 1, mean)) }
rollmean(srate, k = 13, fill = NA)


data = read.csv("./resultat/import_export/history.csv", header = TRUE)

d <- data.frame(x=1:nrow(data), y=ma(data$ZMONT, 4,centre = TRUE), z=data$ZMONT)

plot_ly(d, x=~x) %>%
	add_trace(y=~y, type='scatter', mode='lines') %>%
	add_trace(y=~z, type='scatter', mode='lines')

x <- 1:20
y <- ma(data$ZMONT, 10,centre = TRUE)
z <- rep(0,20)
df <- data.frame(x, y, z)
plot_ly(df, x=~x) %>%
	add_trace(y=~y, type='scatter', mode='lines') %>%
	add_trace(y=~z, type='scatter', mode='lines')

ma(data$ZMONT, 10,centre = TRUE)

mav <- function(x,n){?filter(x,rep(1/n,n), sides=1)} 
library(data.table)
frollmean(x, c(3, 4))
x <- c(1,2,3,4,5,6)
n<-3
cx <- c(0,cumsum(x))
rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n

mav(,2)

for(col in data) {
	cat(col)
}
















