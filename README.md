## Focal Loss
[focal loss](https://arxiv.org/abs/1708.02002) down-weights the well-classified examples. This has the net effect of putting more training emphasis on that data that is hard to classify. In a practical setting where we have a data imbalance, our majority class will quickly become well-classified since we have much more data for it. Thus, in order to insure that we also achieve high accuracy on our minority class, we can use the focal loss to give those minority class examples more relative weight during training.
![](https://github.com/umbertogriffo/focal-loss-keras/blob/master/focal_loss.png)

The focal loss can easily be implemented in Keras as a custom loss function.

## Usage
Compile your model with focal loss as sample:

**Binary**
>model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

**Categorical**
>model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

## References
* The binary implementation is based [@mkocabas' code](https://github.com/mkocabas/focal-loss-keras) 
