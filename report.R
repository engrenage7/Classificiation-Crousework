library(tidyverse)
library(corrplot)
library(skimr)
data <- read.csv("heart_failure.csv")

skim(data)
DataExplorer::plot_bar(data, ncol = 3)
DataExplorer::plot_histogram(data, ncol = 3)
corrplot(cor(data),method='color',tl.cex=0.5)

data %>%
  select(age,ejection_fraction,serum_creatinine,serum_sodium,time,fatal_mi) %>%
  pivot_longer(cols=age:time,names_to="var",values_to="value") %>%
  ggplot(aes(var,value,fill=factor(fatal_mi)))+
  geom_boxplot(alpha=0.5)+
  labs(x="variable",y="value",fill="fatal_mi")+
  theme_bw()

library("data.table")
library("mlr3verse")

set.seed(2024) # set seed for reproducibility
data$fatal_mi <- factor(data$fatal_mi, levels = c("0", "1"))
Heart_task <- TaskClassif$new(id = "Heart_failure",
                              backend = data, # <- NB: no na.omit() this time
                              target = "fatal_mi",
                              positive = "0")
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(Heart_task)

# choose there different model
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
range <- lrn("classif.ranger", predict_type = "prob")
svm <- lrn("classif.svm", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(Heart_task),
  learner    = list(lrn_baseline,lrn_cart,
                    range,svm),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

library(mlr3)
library(mlr3tuning)
library(mlr3learners)
param_set_rpart <- ParamSet$new(list(
  ParamInt$new("minsplit", lower = 1, upper = 10),
  ParamInt$new("maxdepth", lower = 1, upper = 30)
))

param_set_ranger <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("min.node.size", lower = 1, upper = 20),
  ParamInt$new("max.depth", lower = 5, upper = 30)
))

at_rpart <- AutoTuner$new(
  learner = lrn_cart,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = param_set_rpart,
  terminator = trm("evals", n_evals = 20),
  tuner = tnr("grid_search", resolution = 10)
)

at_ranger <- AutoTuner$new(
  learner = range,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.ce"),
  search_space = param_set_ranger,
  terminator = trm("evals", n_evals = 20),
  tuner = tnr("random_search")
)

at_rpart$train(Heart_task)
best_params_rpart <- at_rpart$archive$data[which.min(at_rpart$archive$data$classif.ce), ]
at_ranger$train(Heart_task)
best_params_ranger <- at_ranger$archive$data[which.min(at_rpart$archive$data$classif.ce), ]

learner_rpart_best <- at_rpart$clone()$learner
learner_ranger_best <- at_ranger$clone()$learner

res <- benchmark(data.table(
  task       = list(Heart_task),
  learner    = list(learner_rpart_best,learner_ranger_best),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))