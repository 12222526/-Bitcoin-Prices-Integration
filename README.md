# -Bitcoin-Prices-Integration
library(ggplot2)
library(corrplot)
library(rpart)
library(randomForest)
library(e1071)
library(caret)
library(lattice)
data<-read.csv(file.choose(),sep=',',header=T)
head(data)
str(data)
# Plot time series for closing prices
#Visualize Time Series Data:
ggplot(data, aes(x = as.Date(Date, format="%Y-%m-%d"))) +
  geom_line(aes(y = BTC, color = "Bitcoin")) +
  geom_line(aes(y = NASDAQ, color = "Nasdaq")) +
  geom_line(aes(y = NYSE, color = "NYSE")) +
  geom_line(aes(y = LSE, color = "LSE")) +
  labs(title = "Daily Market Closing Prices", x = "Date", y = "Closing Price") +
  scale_color_manual(values = c("BTC" = "blue", "NASDAQ" = "red", "NYSE" = "green", "LSE" = "purple"))
#Scatter Plot Matrix:
pairs(data[, c("BTC", "BTC_Volume", "NASDAQ", "NYSE", "LSE")], 
      main = "Scatter Plot Matrix")
#Correlation Heatmap:
# Calculate correlation matrix
correlation_matrix <- cor(data[, c("BTC", "BTC_Volume", "NASDAQ", "NASDAQ_Volume", 
                                      "NYSE", "NYSE_Volume", "LSE", "LSE_Volume")], use="complete.obs")

# Plot heatmap
corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
tl.col = "black", tl.srt = 45, addCoef.col = "black")

#Supervised Learning Models: Split the Data
set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$BTC, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
#Model 1: Linear Regression:
linear_model <- lm(BTC ~ BTC_Volume + NASDAQ_Volume + NYSE_Volume + LSE_Volume, data = train_data)
summary(linear_model)

# Predictions
linear_predictions <- predict(linear_model, test_data)
mse_linear <- mean((test_data$BTC - linear_predictions)^2)
print(paste("Linear Regression MSE:", mse_linear))
#Model 2: Decision Tree:
tree_model <- rpart(BTC ~ BTC_Volume + NASDAQ_Volume + NYSE_Volume + LSE_Volume, data = train_data, method = "anova")
rpart(tree_model)

# Predictions
tree_predictions <- predict(tree_model, test_data)
mse_tree <- mean((test_data$BTC - tree_predictions)^2)
print(paste("Decision Tree MSE:", mse_tree))
#Model 4: SVM:
svm_model <- svm(BTC ~ BTC_Volume + NASDAQ_Volume + NYSE_Volume + LSE_Volume, data = train_data)
summary(svm_model)

# Predictions
svm_predictions <- predict(svm_model, test_data)
mse_svm <- mean((test_data$BTC - svm_predictions)^2)
print(paste("SVM Regression MSE:", mse_svm))
#Residual Plot for Model Evaluation:
# Plot residuals for linear regression
ggplot(data.frame(Fitted = linear_predictions, Residuals = test_data$BTC - linear_predictions), 
       aes(x = Fitted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals vs Fitted Values (Linear Regression)", x = "Fitted Values", y = "Residuals")
