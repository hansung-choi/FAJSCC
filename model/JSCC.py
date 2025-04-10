from .common_component import *
from .Encoder import *
from .Decoder import *
import random

class ConvJSCC(nn.Module):
    def __init__(self, model_info):
        super(ConvJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = 2
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp        
        self.chan_type = model_info['chan_type']

        self.Encoder = ConvEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = ConvDecoder(model_info)
        
    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)
        decoder_input = self.channel(encoder_output,SNR_info)
        decoder_output = self.Decoder(decoder_input)

        return decoder_output
    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch

class ResJSCC(nn.Module):
    def __init__(self, model_info):
        super(ResJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = 2
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp        
        self.chan_type = model_info['chan_type']

        self.Encoder = ResEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = ResDecoder(model_info)
        
    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)
        decoder_input = self.channel(encoder_output,SNR_info)
        decoder_output = self.Decoder(decoder_input)

        return decoder_output
    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class SwinJSCC(nn.Module):
    def __init__(self, model_info):
        super(SwinJSCC, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = SwinEncoder(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = SwinDecoder(model_info)
        
    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W

        encoder_output = self.Encoder(x)        
        decoder_input = self.channel(encoder_output,SNR_info)        
        decoder_output = self.Decoder(decoder_input)

        return decoder_output

    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class FAJSCCwoSA(nn.Module): # Content Aware JSCC
    def __init__(self, model_info):
        super(FAJSCCwoSA, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = FAEncoder_woSA(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = FADecoder_woSA(model_info)
        
    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W
        decision = []

        if self.training:
            encoder_output, mask = self.Encoder(x)
            decision.extend(mask)                    
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output, mask = self.Decoder(decoder_input)
            decision.extend(mask)
            return decoder_output, decision          

        else:
            encoder_output = self.Encoder(x)        
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output = self.Decoder(decoder_input)
            return decoder_output


    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class FAJSCCwSA(nn.Module): # Content Aware SNR Adaptive JSCC
    def __init__(self, model_info):
        super(FAJSCCwSA, self).__init__()
        #self.epoch = 0
        self.epoch = nn.Parameter(torch.zeros(1))
        self.color_channel = model_info['color_channel']
        n_block_list = model_info['n_block_list']
        self.n_stage = len(n_block_list)
        self.rcpp = model_info['rcpp'] #rcpp means reverse of channel per pixel
        self.C = int(3*2*(2**self.n_stage)*(2**self.n_stage)*(1/self.rcpp))
        self.cpp = 1/self.rcpp
        self.chan_type = model_info['chan_type']

        self.Encoder = FAEncoder_wSA(model_info)
        self.channel = Channel(self.chan_type)
        self.Decoder = FADecoder_wSA(model_info)
        
    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W
        decision = []

        if self.training:
            encoder_output, mask = self.Encoder(x,SNR_info)
            decision.extend(mask)                    
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output, mask = self.Decoder(decoder_input,SNR_info)
            decision.extend(mask)
            return decoder_output, decision          

        else:
            encoder_output = self.Encoder(x,SNR_info)        
            decoder_input = self.channel(encoder_output,SNR_info)        
            decoder_output = self.Decoder(decoder_input,SNR_info)
            return decoder_output



    def get_epoch(self):
        return self.epoch
    
    def add_epoch(self,number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch        
        
