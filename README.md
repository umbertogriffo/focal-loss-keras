## Focal Loss
[focal loss](https://arxiv.org/abs/1708.02002) down-weights the well-classified examples. This has the net effect of putting more training emphasis on that data that is hard to classify. In a practical setting where we have a data imbalance, our majority class will quickly become well-classified since we have much more data for it. Thus, in order to insure that we also achieve high accuracy on our minority class, we can use the focal loss to give those minority class examples more relative weight during training.
![](https://github.com/umbertogriffo/focal-loss-keras/blob/master/focal_loss.png)

The focal loss can easily be implemented in Keras as a custom loss function.

## Usage
Compile your model with focal loss as sample:

**Binary**
>model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

**Categorical**
>model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)

Alpha is used to specify the weight of different categories/labels, the size of the array needs to be consistent with the number of classes.

**Convert a trained keras model into an inference tensorflow model**

If you use the [@amir-abdi's code](https://github.com/amir-abdi/keras_to_tensorflow) to convert a trained keras model into an inference tensorflow model, you have to serialize nested functions.
In order to serialize nested functions you have to install dill in your anaconda environment as follow:

>conda install -c anaconda dill 

then modify **keras_to_tensorflow.py** adding this piece of code after the imports: 
``` python
import dill
custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=[[.25, .25, .25]]))),
                 'categorical_focal_loss': categorical_focal_loss,
                 'binary_focal_loss': binary_focal_loss}
```                 
and modify the beginning of **load_model** method as follow:
``` python
if not Path(input_model_path).exists():
    raise FileNotFoundError(
        'Model file `{}` does not exist.'.format(input_model_path))
try:
    model = keras.models.load_model(input_model_path, custom_objects=custom_object)
    return model
```

## References
* The binary implementation is based [@mkocabas's code](https://github.com/mkocabas/focal-loss-keras) 
