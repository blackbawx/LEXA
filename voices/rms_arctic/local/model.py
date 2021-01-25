import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *
from util import *


def anneal_tau(steps):
  factor = 1e-5
  return max(0.1, float(np.exp(-1 * factor * steps)))

class AttentionWrapperLexa(nn.Module):
    def __init__(self, rnn_cell, attention_mechanism,
                 score_mask_value=-float("inf")):
        super(AttentionWrapperLexa, self).__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism
        self.score_mask_value = score_mask_value

    def forward(self, query, attention, cell_state, memory,
                processed_memory=None, mask=None, memory_lengths=None, tau=None):

        assert tau is not None

        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)
 
        # Concat input query and previous attention context
        ######### Sai Krishna 15 June 2019 #####################
        if len(query.shape) > 2:
              query = query.squeeze(1)
        #print("Shapes of query and attention: ", query.shape, attention.shape)
        ##########################################################
        cell_input = torch.cat((query, attention), -1)

        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)

        # Alignment
        # (batch, max_time)
        alignment = self.attention_mechanism(cell_output, processed_memory, tau)

        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Normalize attention weight
        latent_classes = torch.argmax(alignment, dim=-1)
        #print("Shape of alignment: ", alignment.shape, amax)
        #sys.exit()

        alignment = F.softmax(alignment,dim=-1)
      
        # Attention context vector
        # (batch, 1, dim)
        attention = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        attention = attention.squeeze(1)

        return cell_output, attention, alignment, latent_classes


class Decoder_TacotronOneSeqwise(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneSeqwise, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim * r, sizes=[256, 128])

class LexaAttention(nn.Module): 
    def __init__(self, dim): 
        super(LexaAttention, self).__init__() 
        self.query_layer = nn.Linear(dim, dim, bias=False) 
        self.tanh = nn.Tanh() 
        self.v = nn.Linear(dim, 1, bias=False) 
 
    def anneal_tau(self, steps):
        factor = 1e-5
        return max(0.1, float(np.exp(-1 * factor * steps)))

    def forward(self, query, processed_memory, tau): 
        """ 
        Args: 
            query: (batch, 1, dim) or (batch, dim) 
            processed_memory: (batch, max_time, dim)
            steps: num_steps 
        """ 

        assert tau is not None

        if query.dim() == 2: 
            # insert time-axis for broadcasting 
            query = query.unsqueeze(1) 
        # (batch, 1, dim) 
        processed_query = self.query_layer(query) 
 
        # (batch, max_time, 1) 
        alignment = self.v(self.tanh(processed_query + processed_memory)/ tau ) 

        # (batch, max_time) 
        return alignment.squeeze(-1)


class Decoder_Lexa(Decoder_TacotronOneSeqwise):
    def __init__(self, in_dim, r, num_encoder_states):
        super(Decoder_Lexa, self).__init__(in_dim, r)

        self.attention_rnn = AttentionWrapperLexa(
            nn.GRUCell(256 + 128, 256),
            LexaAttention(256)
            )

        self.num_encoder_states = num_encoder_states

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None, tau=None):

        assert tau is not None
        #print("The value of tau in decoder: ", tau)

        B = encoder_outputs.size(0)

        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None
        
        if inputs is not None:
            # Grouping multiple frames if necessary
            if inputs.size(-1) == self.in_dim:
                inputs = inputs.view(B, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == self.in_dim * self.r
            T_decoder = inputs.size(1)


        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_())

        # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []
        classes = []

        t = 0

        current_input = initial_input
        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else inputs[t - 1]
            # Prenet
            ####### Sai Krishna Rallabandi 15 June 2019 #####################
            #print("Shape of input to the decoder prenet: ", current_input.shape)
            if len(current_input.shape) < 3:
               current_input = current_input.unsqueeze(1)
            #################################################################
 
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment, latent_classes = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory, mask=mask, tau=tau)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = decoder_input
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]
            classes += [latent_classes]

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        assert greedy or len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        classes = torch.cat(classes, dim=0)
        #print("Shape of classes: ", classes.shape)

        # Entropy
        hist = classes.float().cpu().histc(bins=self.num_encoder_states, min=-0.5, max=1.5)
        probs = hist.masked_select(hist > 0) / len(classes)
        entropy = - (probs * probs.log()).sum().item()
        
        return outputs, alignments, entropy, ' '.join(str(k) for k in classes.cpu().numpy().tolist())


class TacotronOneSeqwise(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwise, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)


# https://github.com/mkotha/WaveRNN/blob/master/layers/downsampling_encoder.py
class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        prev_channels = channels
        total_scale = 1
        pad_left = 0
        self.skips = []
        for stride, ksz, dilation_factor in layer_specs:
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)

            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            self.skips.append(skip)
            total_scale *= stride
        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)

    def forward(self, samples):
        x = samples.transpose(1,2) #.unsqueeze(1)
        #print("Shape of input: ", x.shape)
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec
            #print(i, "Stride, ksz, DF and shape of input: ", stride, ksz, dilation_factor, x.shape)
            x1 = conv_wide(x)
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = conv_1x1(x2)
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        return x.transpose(1, 2)


class MelVQVAEBaseline(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(MelVQVAEBaseline, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        # Stride, KernelSize, DilationFactor
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(mel_dim, encoder_layers)
        self.quantizer = quantizer_kotha(n_channels=1, n_classes=200, vec_len=80)
        self.upsample_scales = [2,4,2,4]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.decoder_lstm = nn.LSTM(80, 128, bidirectional=True, batch_first=True)
        self.decoder_fc = nn.Linear(80,256)
        self.mel_dim = mel_dim  
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)

    def forward(self, mel):
        B = mel.shape[0]
        encoded = self.encoder(mel)
        #print("Shape of encoded: ", encoded.shape)
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(encoded.unsqueeze(2))
        #print("Shape of quantized: ", quantized.shape)
        quantized = quantized.squeeze(2)
        #upsampled = self.upsample_network(quantized)
        #outputs, hidden = self.decoder_lstm(upsampled)
        #outputs =  self.decoder_fc(outputs)
        decoder_input = torch.tanh(self.decoder_fc(quantized))
        mel_outputs, alignments = self.decoder(decoder_input, mel)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        #print("Shape of outputs: ", mel_outputs.shape)

        return mel_outputs, alignments, vq_penalty.mean(), encoder_penalty.mean(), entropy

class TacotronLexa(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):

        super(TacotronLexa, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.encoder = Encoder_TacotronOne(mel_dim)

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(mel_dim, encoder_layers)


    def forward(self, targets=None, input_lengths=None):

        B = targets.size(0)

        encoder_outputs = self.encoder(targets)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


class DownsamplingEncoderStrict(nn.Module): 
    """ 
        Input: (N, samples_i) numeric tensor 
        Output: (N, samples_o, channels) numeric tensor 
    """ 
    def __init__(self, channels, layer_specs, input_dim = 80, use_batchnorm=0): 
        super().__init__() 
 
        self.convs_wide = nn.ModuleList() 
        self.convs_1x1 = nn.ModuleList() 
        self.layer_specs = layer_specs 
        prev_channels = input_dim
        total_scale = 1 
        pad_left = 0 
        self.skips = [] 
        for stride, ksz, dilation_factor in layer_specs: 
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor, padding=int((ksz-1)/2)) 
            wsize = 2.967 / math.sqrt(ksz * prev_channels) 
            conv_wide.weight.data.uniform_(-wsize, wsize) 
            conv_wide.bias.data.zero_() 
            self.convs_wide.append(conv_wide) 
 
            conv_1x1 = nn.Conv1d(channels, channels, 1) 
            conv_1x1.bias.data.zero_() 
            self.convs_1x1.append(conv_1x1) 
 
            prev_channels = channels 
            skip = (ksz - stride) * dilation_factor 
            pad_left += total_scale * skip 
            self.skips.append(skip) 
            total_scale *= stride 
        self.pad_left = pad_left 
        self.total_scale = total_scale 
 
        self.final_conv_0 = nn.Conv1d(channels, channels, 1) 
        self.final_conv_0.bias.data.zero_() 
        self.final_conv_1 = nn.Conv1d(channels, channels, 1) 
        self.batch_norm = nn.BatchNorm1d(channels, momentum=0.9) 
        self.use_batchnorm = use_batchnorm

    def forward(self, samples):
        x = samples.transpose(1,2) #.unsqueeze(1)
        #print("Shape of input: ", x.shape)
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec
            #print(i, "Stride, ksz, DF and shape of input: ", stride, ksz, dilation_factor, x.shape)
            x1 = conv_wide(x)
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = conv_1x1(x2)
            #if i == 0:
            #    x = x3
            #else:
            #    x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            x = x3
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        #print("Shape of output: ", x.shape)
        if self.use_batchnorm:
           return self.batch_norm(x).transpose(1, 2)
        return x.transpose(1,2)

class LexatronDownsampled(TacotronLexa):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):

        super(LexatronDownsampled, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoderStrict(mel_dim, encoder_layers,use_batchnorm=1)
        self.mel_fc = nn.Linear(mel_dim, 256)
        self.decoder = Decoder_Lexa(mel_dim, r, 68)

    def forward(self, targets=None, input_lengths=None, steps=None):

        assert steps is not None

        B = targets.size(0)

        encoder_outputs = self.encoder(targets)
        encoder_outputs = torch.tanh(self.mel_fc(encoder_outputs))
        #print("Shape of targets and encoder outputs: ", targets.shape, encoder_outputs.shape)

        memory_lengths = None
        tau = anneal_tau(steps)
        #print("The value of tau is ", tau)
        mel_outputs, alignments, entropy, classes = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths, tau=tau)
        #print("Entropy is ", entropy)
        #sys.exit()

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments, tau, entropy, classes


