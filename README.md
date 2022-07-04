# TCN_Seq2Seq
Implementation of different TCN based Sequence-to-Sequence models. 

The project includes wrapper classes for the models that add some additional 
features like simple saving and loading of the trained models and automated 
hyperparameter tuning. Also, there is functionality for basic data preprocessing 
as well as saving and loading the preprocessing configurations.  
The project is written in Python 3.8 using Tensorflow 2.5.0

TCN_Seq2Seq is on PyPI, so you can use `pip` to install it.

```bash
pip install --upgrade pip
pip install tcn-sequence-models
```

### Models
Two models exist:
#### 1. TCN-TCN model. The structure of the TCN-TCN_Model can be seen below.


There are two versions of this model, an autoregressive model and a 
none-autoregressive model.

Autoregressive model:
For training, teacher-forcing is used for the decoder. In addition to other decoder 
input data that is defined by the user, always the ground truth of the previous time 
step is added as an additional feature. For example at position x(n+1) the ground 
truth at position x(n) is added as an additional input.
During inference, the predictions of the previous time step is used instead.
The autoregressive model can be trained fast in a parallel way. However, inference 
is slow because the decoder needs to be passed m times when the output length is m.

None-regressive model:
For the none-regressive model only the input features are used for the decoder that 
the user provides. If the user does not provide any features for the decoder, a 
dummy feature is created with all 1s so that the decoder has some input. Without any 
input the model would not work. This model is fast during training and inference.

##### Encoder
The encoder consists of a TCN block. 

##### Decoder
The Decoder architecture is as follows:  
First a TCN stage is used to encoder the decoder input data.
After that multi-head cross attention is applied the the TCN output and the
encoder output.
Then another TCN stage is applied. The input of this TCN stage is a
concatenation of the output of the first Decoder-TCN and the output of the
cross attention.
The last stage is the prediction stage (a block of dense layers) that then
makes the final prediction.

![Model plot](./images/TCN-TCN.jpg)

#### 2. TCN-RNN model.
The architecture of the TCN-RNN model can be seen in the following image:
![Model plot](./images/TCN-GRU.jpg)

### TCN blocks
The TCN blocks use as many layers as needed to get a connection from first timestep's 
input to last timestep's output. The padding mode can be set by the user. Only for 
the decoder of the autoregressive TCN-TCN model padding will always be causal. The 
picture below shows 'causal' padding.
![TCN plot](./images/TCN.jpg)

### Inputs
This model expects input sequences for the encoder and optionally for the decoder.

### Examples
For examples, please refer to the notebooks.

