import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResidualCNN(nn.Module):
    def __init__(self, out_channels=512):
        super(ResidualCNN, self).__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, out_channels)

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            ResidualBlock(out_c),
            ResidualBlock(out_c)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x.permute(0, 2, 3, 1)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state, encoder_projections=None):
        if encoder_projections is None:
            encoder_projections = self.U(features)
        hidden_proj = self.W(hidden_state).unsqueeze(1)
        attention_weights = torch.softmax(self.A(torch.tanh(encoder_projections + hidden_proj)).squeeze(2), dim=1)
        context = torch.einsum('bn,bnd->bd', attention_weights, features)
        return context, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, attention_dim):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)

    def forward(self, features, captions):
        batch_size = features.size(0)
        encoder_dim = features.size(-1)
        features = features.view(batch_size, -1, encoder_dim)
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(features)
        seq_length = captions.size(1) - 1
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        encoder_projections = self.attention.U(features)

        for t in range(seq_length):
            context, _ = self.attention(features, h, encoder_projections)
            gate = torch.sigmoid(self.f_beta(h))
            lstm_input = torch.cat((embeddings[:, t], gate * context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs[:, t] = self.fc(self.dropout(h))
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, attention_dim, device="cuda"):
        super(CNNtoRNN, self).__init__()
        self.encoder = ResidualCNN(out_channels=encoder_dim)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, encoder_dim, attention_dim)
        self.device = device

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=20):
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(self.device))
            features = features.view(1, -1, features.size(-1))
            h, c = self.decoder.init_hidden_state(features)
            word_idx = vocabulary.stoi["<SOS>"]
            caption = []
            encoder_projections = self.decoder.attention.U(features)

            for _ in range(max_length):
                embed = self.decoder.embedding(torch.tensor([word_idx]).to(self.device))
                context, _ = self.decoder.attention(features, h, encoder_projections)
                gate = torch.sigmoid(self.decoder.f_beta(h))
                lstm_input = torch.cat((embed, gate * context), dim=1)
                h, c = self.decoder.lstm_cell(lstm_input, (h, c))
                output = self.decoder.fc(h)
                predicted = output.argmax(1).item()
                if predicted == vocabulary.stoi["<EOS>"]: break
                caption.append(vocabulary.itos[predicted])
                word_idx = predicted
        self.train()
        return " ".join(caption)
