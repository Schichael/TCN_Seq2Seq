# TCN_Seq2Seq
**[Work in Progress]**  

The entire project is written in Python 3.8 using Tensorflow 2.5.0

### TCN-Seq2Seq Model
TCN-based sequence-to-sequence model for time series forecasting.  
Influenced by the Transformer model, it uses Multi-Headed Encoder-Decoder-Attention to 
connect encoder and decoder.  
Instead of Self-Attention as used in the Transformer architecture, this model uses TCN 
blocks. Additional positional encoding  is not necessary since a TCN stage performs 
temporal encoding implicitly.  
Also, this model does not use auto-correlation in the sense that the t-1th prediction 
is not fed as input to compute the t-th prediction. This feature is planned to be 
implemented in the future.  

### Input
This model expects input sequences both for the encoder and for the decoder.  
The timesteps of the decoder inputs correspond to the timesteps of the made predicitons.
Inputs: [encoder_input, decoder_input]

### Encoder
The encoder consists of n TCN blocks stacked on top of each other.
Optionally, residual connections can be used between consecutive TCN blocks.

### Decoder
The decoder consists of n stages.
Each stage begins with a TCN block. It follows an Encoder-Decoder
Multi-Head-Attention layer (multiplicative attention) that connects the output of the 
encoder with the output of the TCN block. The output of the TCN block and the
attention layer are then summed and normalized.  
Optionally, residual connections can be used between the output of one
stage and the output of the TCN block of the gollowing stage.
After the last stage, another TCN block is added that generates the output.  
> **Note**: The output of the decoder is NOT the prediction. After the decoder, an 
> output stage (e.g. a feed forward NN) still needs to be added to compute the prediction.
> The complete TCN_Seq2Seq model includes an output stage

### TCN blocks
The TCN blocks use as many layers as needed to get a connection from first timestep's 
input to last timestep's output. Padding can be set by the user. (usually 'causal' or 'same')



