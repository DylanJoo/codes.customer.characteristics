import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, dim, latent_dim, entity_size, hidden_dims=(512, 256, 128), dropout_rate=0.1):
        super(AutoEncoder, self).__init__()

        assert latent_dim < hidden_dims[2], 'incorrect AE dimension setting'
        self.embeddings = nn.Embedding(entity_size, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dims[0]), nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]), nn.Tanh(),
            nn.Linear(hidden_dims[2], latent_dim),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]), nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[1]), nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.Tanh(),
            nn.Linear(hidden_dims[0], dim),
        )
        self.loss_fct = nn.MSELoss()

    def forward(self, user_ids, user_inputs):
        # user representation
        user_embeds = self.embeddings(user_ids)
        src_x = user_inputs

        # interacrtion representation
        latent_x = self.encoder(src_x)
        _latent_x = self.dropout(latent_x)
        tgt_x = self.decoder(self.dropout(latent_x) + user_embeds)

        # FP loss
        loss = self.loss_fct(tgt_x, src_x)

        return {'user_embeds': user_embeds, 'latent_embeds': latent_x, 'logits': tgt_x, 'loss': loss}

