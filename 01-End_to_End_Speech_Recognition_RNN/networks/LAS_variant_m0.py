import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random

LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()

    letter2index = {c: i for i, c in enumerate(letter_list)}
    index2letter = {i: c for i, c in enumerate(letter_list)}

    return letter2index, index2letter
    

def transform_index_to_letter(batch_indices, stopping_idx):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    for indices in batch_indices:
        transcript = ""
        for i in indices:
            if i == stopping_idx:
                break
            else:
                transcript += LETTER_LIST[i]
        transcripts.append(transcript)
    return transcripts
        
# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        # input_dim is multiplied by 2 because of the reshape in forward
        self.blstm = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        ''' 
        Since the input to the initial BLSTM layer is a packed sequence, the output is also packed, 
        which means that the input to this pBLSTM layer is a packed sequence.
        So, we need to pad the input first (use pad_packed_sequence)
        '''
        padded_input, input_lengths = pad_packed_sequence(x, batch_first=True)
        padded_input = padded_input[:, :padded_input.shape[1]//2*2, :]
        torch.cuda.empty_cache()
        
        # reduce the time resolution by a factor of 2 -- concatenate every 2 previous states
        reduced_input = padded_input.reshape(padded_input.shape[0], padded_input.shape[1] // 2, padded_input.shape[2] * 2)
        reduced_lengths = torch.div(input_lengths, 2, rounding_mode='floor').to(padded_input.device)

        # pack the reduced input
        packed_input = pack_padded_sequence(reduced_input, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, _ = self.blstm(packed_input)

        return out

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # Define the blocks of pBLSTMs
        # input_dim is multiplied by 2 because of bidirectional.
        self.pBLSTMs = nn.Sequential(
            pBLSTM(input_dim=encoder_hidden_dim * 2, hidden_dim=encoder_hidden_dim),
            pBLSTM(input_dim=encoder_hidden_dim * 2, hidden_dim=encoder_hidden_dim),
            pBLSTM(input_dim=encoder_hidden_dim * 2, hidden_dim=encoder_hidden_dim),
            # Optional: dropout
            # ...
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(encoder_hidden_dim * 2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim * 2, key_value_size)

    def forward(self, x, x_len):
        """
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)

        # pass the input to the pyramidal LSTM layers
        out, _ = self.lstm(packed_input)
        out = self.pBLSTMs(out)
        torch.cuda.empty_cache()

        out, out_lengths = pad_packed_sequence(out, batch_first=True)
        torch.cuda.empty_cache()

        key = self.key_network(out)
        value = self.value_network(out)

        return key, value, out_lengths

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    Here are different ways to compute attention and context:
    1. Dot-product attention
        energy = bmm(key, query) 
        # Optional: Scaled dot-product by normalizing with sqrt key dimension
        # Check "attention is all you need" Section 3.2.1
    * 1st way is what most TAs are comfortable with, but if you want to explore...
    2. Cosine attention
        energy = cosine(query, key) # almost the same as dot-product xD 
    3. Bi-linear attention
        W = Linear transformation (learnable parameter): d_k -> d_q
        energy = bmm(key @ W, query)
    4. Multi-layer perceptron
        # Check "Neural Machine Translation and Sequence-to-sequence Models: A Tutorial" Section 8.4
    
    After obtaining unnormalized attention weights (energy), compute and return attention and context, i.e.,
    energy = mask(energy) # mask out padded elements with big negative number (e.g. -1e9)
    attention = softmax(energy)
    context = bmm(attention, value)

    5. Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self, dropout=None):
        super(Attention, self).__init__()
        # Optional: dropout
        if not dropout is None:
            self.dropout = dropout

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        
        """
        # print(f"query size: {query.shape}")
        # print(f"key size: {key.shape}")
        # Scaled dot-product by normalizing with sqrt key dimension
        energy = torch.bmm(query.unsqueeze(1), key.permute(0, 2, 1)).squeeze(1).div_(np.sqrt(key.shape[2]))
        # print(f"energy size: {energy.shape}")

        # apply mask and calculate attention
        energy.masked_fill_(mask, -1e9)
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        # print(f"attention size: {attention.shape}")
        # print(f"context size: {context.shape}")

        return attention, context
        # we return attention weights for plotting (for debugging)

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=letter2index['<eos>'])
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)

        self.attention = Attention()     
        self.vocab_size = vocab_size
        # Optional: Weight-tying
        self.character_prob = nn.Linear(key_value_size*2, vocab_size) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight

    def forward(self, key, value, encoder_len, y=None, mode='train', teacher_forcing_rate=1.0):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 
        '''

        B, key_seq_max_len, key_value_size = key.shape
        device = key.device

        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # TODO: Create the attention mask here (outside the for loop rather than inside) to aviod repetition
        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= encoder_len.unsqueeze(1)
        mask = mask.to(device)
        
        predictions = []
        # This is the first input to the decoder
        # What should the fill_value be?
        prediction = torch.full((B,), fill_value=letter2index['<sos>'], device=device)
        # The length of hidden_states vector should depend on the number of LSTM Cells defined in init
        # The paper uses 2
        hidden_states = [None, None] 
        
        # TODO: Initialize the context
        context = value.mean(axis=1)

        attention_plot = [] # this is for debugging

        for i in range(max_len):
            if mode == 'train':
                # TODO: Implement Teacher Forcing
                
                if teacher_forcing_rate > random.random():
                    if i == 0:
                        # This is the first time step
                        # Hint: How did you initialize "prediction" variable above?
                        first_teacher = torch.full((B,), fill_value=letter2index['<sos>'], device=device)
                        char_embed = self.embedding(first_teacher)
                    else:
                        # Otherwise, feed the label of the **previous** time step
                        char_embed = char_embeddings[:, i-1, :]
                else:
                    # embedding of the previous prediction
                    if i == 0:
                        char_embed = self.embedding(prediction)
                    else:
                        char_embed = self.embedding(prediction.argmax(dim=-1))
                

            else:
                # embedding of the previous prediction
                if i == 0:
                    char_embed = self.embedding(prediction)
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            # what vectors should be concatenated as a context?
            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1])
            # What then is the query?
            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            attention, context = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0].detach().cpu())
            
            # What should be concatenated as the output context?
            output_context = torch.cat([query, context], dim=1)

            # print(output_context.shape)
            # print(self.character_prob.weight.shape)

            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        # print(f"prediction.shape={prediction.shape}")
        # print(f"predictions.shape={predictions.shape}")
        return predictions, attentions

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)

    def forward(self, x, x_len, y=None, mode='train', teacher_forcing_rate=1.0):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode, teacher_forcing_rate=teacher_forcing_rate)
        return predictions, attentions