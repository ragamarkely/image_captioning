import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
    def forward(self, features, captions):
        # Input sizes.
        batch_size = features.size(0)
        sequence_length = captions.size(1)
        
        # Set up device.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set up inputs.
        features.unsqueeze_(1)
        captions = self.embedding(captions[:, :-1])  # Truncate caption to maintain the same length.
        inputs = torch.cat((features, captions), dim=1)
        
        # Set up hidden and cell states.
        hidden_state = torch.zeros(1, batch_size, self.hidden_size).to(device)
        cell_state = torch.zeros(1, batch_size, self.hidden_size).to(device)
        
        # LSTM layer.
        # Output shape: batch_size, sequence_length, hidden_size
        outputs, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))
        
        # FC layer.
        outputs = self.fc(outputs)  # output shape = batch_size, sequence_length, vocab_size
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Input sizes.
        batch_size = inputs.size(0)
        
        # Set up device.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Output
        length = 0
        outputs = []
        
        # Set up hidden and cell states.
        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        while length < max_len:
            # LSTM layer.
            output, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))
            # FC layer.
            output = self.fc(output)  # Shape: batch_size x seq_length x vocab_size = N x 1 x vocab_size
            output.squeeze_(1)  # Shape: N x vocab_size
            
            # Most likely word
            _, word_ix = torch.max(output, dim=1)
            outputs.append(word_ix.item())
            if word_ix == 1:
                break
            
            length += 1
            
            # Use output as input.
            inputs = self.embedding(word_ix)  # Shape: batch_size x embed_size
            inputs.unsqueeze_(1)  # Shape: batch_size x 1 (seq_length) x embed_size
            
        return outputs