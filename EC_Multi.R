############################
## Analisis Discriminante
##
###########################

rm(list=ls()) ## borrar variables del ambiente R
library(readxl)
setwd("/Users/andreacamilaacosta/Downloads")
# Leer un archivo Excel
data <- read_excel("BDTrabajoFinal.xlsx")
head(data)
dim(data)
table(data[,3]) #cantidad de cad y normal
table(data[,2]) #cantidad de hombres y mujeres
colnames(data)[which(names(data) == "Region RWMA")] <- "Region_RWMA"
colnames(data)[which(names(data) == "EF-TTE")] <- "EF_TTE"
data$gender_binary <- ifelse(data$Sex == "Fmale", 1,
                             ifelse(data$Sex == "Male", 0, NA))

sum(data$gender_binary == 1)
sum(data$gender_binary == 0)
data[1:100,27]
library(caret)

# Definir el porcentaje para los datos de prueba
set.seed(123)  # Semilla
index <- createDataPartition(data$Cath, p = 0.3, list = FALSE)  # 30% para prueba

# Dividir los datos
data_prueba <- data[index, ]   # Datos de prueba
dim(data_prueba)
data_entrenamiento <- data[-index, ]  # Datos de entrenamiento
dim(data_entrenamiento)
# Verificar proporciones
table(data_prueba$Cath)
head(data_prueba)

table(data_entrenamiento$Cath)
data_entrenamiento$Cath<-as.factor(data_entrenamiento$Cath)
library(MASS)

str(data_entrenamiento) 

data_entrenamiento=data_entrenamiento[,-1]
data_entrenamiento=data_entrenamiento[,-1]

library(MASS)


lda<-lda(Cath ~ FH + Region_RWMA+ Age + Weight + Length + BMI + BP  + PR 
         +FBS +CR + TG + LDL + HDL + BUN + ESR + HB + K + Na
         +WBC + Lymph + Neut + PLT + EF_TTE + gender_binary, data=data_entrenamiento)
lda

lda # Aqu? salen los coeficientes de la/s funci?n/es discriminate/s

lda$counts # Este es el n?mero total de observaciones por grupo - reales

lda.p=predict(lda) ### Proceso de Predicci?n
lda.p$class ###  N?mero de observaciones que lda clasific? por grupo.
table(lda.p$class)
Tabla=table(data_entrenamiento$Cath,lda.p$class)
Tabla

diag(prop.table(Tabla, 1)) # Porcentajes de aciertos por grupo en la muestra de adiestramiento
sum(diag(prop.table(Tabla)))

### Encontrando el porcentaje de aciertos total

prop.table(Tabla)

sum(diag(prop.table(Tabla))) # Porcentaje de acierto total en la muestra de adiestramiento

plot(lda,col="blue") 
plot(lda)
### La funci?n de discriminaci?n se le aplica a la muestra Experimental
reales <- reales <- data_prueba$Cath
data_prueba<-data_prueba[,-1]
head(data_prueba)
data_prueba<-data_prueba[,-1]
data_prueba<-data_prueba[,-1]
pred=predict(lda,data_prueba)
pred

pred$class

data.frame(pred$class)


### Datos de Experimental ahora con los grupos a los cuales pertenecen los individuos

data_prueba$Grupo_Estimado=with(data_prueba,pred$class)
data_prueba
table(reales)
table(pred$class)
Tabla=table(reales,pred$class)
Tabla
diag(prop.table(Tabla, 1))
sum(diag(prop.table(Tabla)))


lda_data <- data.frame(Grupo = data_entrenamiento$Cath,
                       LD1 = lda.p$x[, 1])  # Primer y único discriminante

# Graficar densidades
ggplot(lda_data, aes(x = LD1, fill = Grupo)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribución del Primer Discriminante",
       x = "Primer Discriminante",
       y = "Densidad") +
  theme_minimal()

plot(lda)
