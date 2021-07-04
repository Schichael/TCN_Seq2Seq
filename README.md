# TCN_Seq2Seq
Sequence to sequence TCN-based model for time series forecasting.  
        Influenced by the Transformer model, it used Multi-Headed  
        Encoder-Decoder-attention to connect encoder and decoder. Instead of  
        Self-attention as used in the Transformer architecture, this model uses  
        TCN stages. Additional positional encoding  is not necessary since a TCN  
        stage performs temporal encoding implicitly. Another difference to the  
        Transformer is that multi headed attention is computed differently. Here,  
        the inputs(query, key, value) are not split before feeding the attention  
        heads. This is done due to the usually low dimensionality of the inputs.  
        Also, this model does not use auto-correlation in the sense that the t-1th  
        prediction is not fed as input to compute the t-th prediction.  
        Inputs: [encoder_input, decoder_input]  
