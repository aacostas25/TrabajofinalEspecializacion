##### Descriptivas
#### Algunas estadisticas descriptivas
datos$Sex <- as.factor(datos$Sex)
datos$Cath <- as.factor(datos$Cath)
table(datos$Sex, datos$Cath)


library(ggplot2)
ggplot(datos, aes(x=Sex, fill=Cath)) + 
  geom_bar(position="dodge") + 
  labs(title="Distribución de Género por Grupo", x="Género", y="Frecuencia") + theme_minimal()

ggplot(datos, aes(x=LDL, y=HDL, color=Cath)) + 
  geom_point(alpha=0.7) + 
  labs(title="Relación entre LDL y HDL por Grupo", x="LDL", y="HDL") + 
  theme_minimal()

ggplot(datos, aes(x=Obesity, fill=Cath)) + 
  geom_bar(position="dodge") + 
  labs(title="Distribución de Obsesidad por Grupo", x="Grupo", y="Frecuencia") + 
  theme_minimal()

ggplot(datos, aes(x=as.factor(FH), fill=Cath)) + 
  geom_bar(position="dodge") + 
  labs(title="Distribución de Historial Familiar por Grupo", x="Grupo", y="Frecuencia") + 
  theme_minimal()

ggplot(datos, aes(x=Cath, y=LDL, fill=Cath)) + 
  geom_boxplot(alpha=0.7) + 
  labs(title="Boxplot de LDL por Grupo", x="Grupo", y="LDL") + 
  theme_minimal()

ggplot(datos, aes(x=Cath, y=HDL, fill=Cath)) + 
  geom_boxplot(alpha=0.7) + 
  labs(title="Boxplot de HDL por Grupo", x="Grupo", y="HDL") + 
  theme_minimal()


