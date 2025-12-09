import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Freeze Inception weights if we are not fine-tuning
        if not self.train_CNN:
            for name, param in self.inception.named_parameters():
                if "fc.weight" in name or "fc.bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, images):
        features = self.inception(images)
        
        # Inception v3 returns (logits, aux_logits) during training
        if hasattr(features, 'logits'):
            features = features.logits
        elif isinstance(features, tuple):
            features = features[0]
            
        # BatchNorm requires more than 1 value per channel if in training mode
        # If batch size is 1, we must switch to eval mode for this layer or skip it
        if features.size(0) > 1:
            return self.dropout(self.relu(self.bn(features)))
        else:
            # Skip BatchNorm for batch size 1 (inference/eval)
            return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(self.dropout(hiddens))
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

    def caption_image_beam_search(self, image, vocabulary, max_length=50, beam_size=3):
        """
        Generates a caption using beam search.
        """
        k = beam_size
        
        with torch.no_grad():
            # 1. Get image features from Encoder
            # Shape: (1, embed_size) -> (1, 1, embed_size) for LSTM input
            encoder_out = self.encoder(image).unsqueeze(0)
            
            # 2. Initialize the beam
            # Each candidate is a tuple: (list_of_token_indices, log_prob_score, lstm_states)
            # We start with an empty sequence (or just the image processed)
            # Note: Our model predicts Word1 from Image.
            
            # Initial step: Feed image features to LSTM
            hiddens, states = self.decoder.lstm(encoder_out, None)
            outputs = self.decoder.linear(hiddens.squeeze(0)) # (1, vocab_size)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            
            # Get top k starting words
            topk_probs, topk_ids = log_probs.topk(k)
            
            candidates = []
            for i in range(k):
                word_idx = topk_ids[0][i].item()
                score = topk_probs[0][i].item()
                candidates.append(([word_idx], score, states))
            
            # 3. Beam Search Loop
            for _ in range(max_length - 1):
                all_candidates = []
                
                for seq, score, state in candidates:
                    # If sequence ended with EOS, keep it as is
                    if seq[-1] == vocabulary.stoi["<EOS>"]:
                        all_candidates.append((seq, score, state))
                        continue
                    
                    # Prepare input for next step (embedding of last word)
                    last_word_idx = torch.tensor([seq[-1]]).to(image.device)
                    embed = self.decoder.embed(last_word_idx).unsqueeze(0) # (1, 1, embed_size)
                    
                    # Forward pass
                    hiddens, new_state = self.decoder.lstm(embed, state)
                    outputs = self.decoder.linear(hiddens.squeeze(0))
                    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                    
                    # Get top k next words
                    topk_probs, topk_ids = log_probs.topk(k)
                    
                    for i in range(k):
                        word_idx = topk_ids[0][i].item()
                        word_prob = topk_probs[0][i].item()
                        
                        new_seq = seq + [word_idx]
                        new_score = score + word_prob
                        all_candidates.append((new_seq, new_score, new_state))
                
                # Select top k candidates globally
                ordered = sorted(all_candidates, key=lambda t: t[1], reverse=True)
                candidates = ordered[:k]
                
                # Stop if all candidates are finished
                if all(c[0][-1] == vocabulary.stoi["<EOS>"] for c in candidates):
                    break
            
            # Return best sequence
            best_seq = candidates[0][0]
            return [vocabulary.itos[idx] for idx in best_seq if idx != vocabulary.stoi["<EOS>"]]
