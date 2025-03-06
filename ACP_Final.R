####################################
# Analisis de componentes principales
#####################################

rm(list=ls()) ## borrar variables del ambiente R
library(readxl)
setwd("/Users/andreacamilaacosta/Downloads")
# Leer un archivo Excel
data <- read_excel("Z-Alizadeh-sani-dataset.xlsx")
head(data)
dim(data)
table(data[,3]) #cantidad de cad y normal
table(data[,2]) #cantidad de hombres y mujeres
data<-data[,-1]
# Seleccionar solo columnas numéricas
datos1 <- data[, sapply(data, is.numeric)]
head(datos1)
datos1<-datos1[,-1]
datos1<-datos1[,-1]
head(datos1)
sapply(datos1, mean)  ## media de todas las variables


sapply(datos1, var)    ## varianza de todas las variables


sapply(datos1, sd)     ## Desviaci?n est?ndar


sapply(datos1, range)  ## Rango 



S<-round(cov(datos1),3)     ##  Matriz de covarianza
S


Rho<-round(cor(datos1),3)     ##  Matriz de correlaci?n
Rho


library(corrplot)

corrplot.mixed(Rho) 


acp<- prcomp(datos1, scale = FALSE) 
acp

names(acp)     ### salidas del an?lisis

acp$sdev       ### Raiz Cuadrada de las lambdas o Desviaci?n est?ndar de Yi
acp$center     ### media de todas las variables ORIGINALES
acp$scale      ### desviaci?n est?ndar de las variables ORIGINALES
acp$x          ### Coordenadas de los individuos
acp$rotation   ### Vectores propios normalizados


sd_lambda<-acp$sdev
sd_lambda


lambda<-sd_lambda**2
lambda

sum(lambda)


MediasX<-acp$center
MediasX

sdX<-acp$scale
sdX


summary(acp) ### Resumen del ACP


##### Coeficientes de las Componentes Principales - vectores

vectores<-acp$rotation  ###  A la matriz se le puede llamar P
vectores

P<-acp$rotation
P

X<-as.matrix(datos1)
X
P<-vectores
P

CP<-X%*%P
CP

#####  screeplot

screeplot(acp)  
screeplot(acp,type="lines") 

A<-cbind(CP[,1],CP[,2])
A
plot(A,  xlab="Componente Principal 1",   ylab="Componente Principal 2") ### graficando la primera y segunda componente
identify(CP[,1],CP[,2])
a<-data[148,]
a<-data[261,]
a<-data[294,]
#text(A,row.names(A),pos=1)  ### colocando el nombre a los puntos
# Separar las etiquetas para colorear
grupos <- data$Cath  # Suponiendo que 'Cath' es la columna con los grupos "Cad" y "Normal"

# Crear un vector de colores basado en los grupos
colores <- ifelse(grupos == "Cad", "red", "blue")

# Graficar las dos primeras componentes principales con colores según los grupos
plot(CP[, 1], CP[, 3], 
     col = colores,
     xlab = "Componente Principal 1", 
     ylab = "Componente Principal 2",
     main = "Gráfico de Componentes Principales")

# Agregar una leyenda
legend("bottomright", legend = c("Cad", "Normal"), col = c("red", "blue"), pch = 19)
coord<-acp$x

coord
plot(coord[,1],coord[,2],col = colores)


biplot(acp)
acp<-prcomp(X, scale = T)
acp

biplot(acp)
