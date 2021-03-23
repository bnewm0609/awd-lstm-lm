import torch
import torch.nn as nn


class AWD_LSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        model_path = "WT2.pt"
        old_model, _, _ = torch.load(model_path, map_location=args['device'])

        # this model was trained with an older version of pytorch so
        # it doesn't work. We're just going to copy the weights and
        # hope for the best

        # encoder
        vocab_size, lstm_io_size = old_model.encoder.weight.shape
        self.encoder = torch.nn.Embedding(vocab_size, lstm_io_size)
        self.encoder.load_state_dict(old_model.encoder.state_dict())

        # decoder
        self.decoder = torch.nn.Linear(lstm_io_size, vocab_size)
        self.decoder.load_state_dict(old_model.decoder.state_dict())

        # lstms
        lstm_hidden_size = 1150
        lstms = [nn.LSTM(lstm_io_size, lstm_hidden_size, batch_first=True),
                nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True),
                nn.LSTM(lstm_hidden_size, lstm_io_size, batch_first=True)]

        for i in range(len(lstms)):
            old_state_dict = old_model.rnns[i].module.state_dict()
            print(old_state_dict)
            old_state_dict['weight_hh_l0'] = old_state_dict['weight_hh_l0_raw'] # used for weight dropout in original model
            del old_state_dict['weight_hh_l0_raw']
            lstms[i].load_state_dict(old_state_dict)

        self.rnns = nn.ModuleList(lstms)
        # self.batch_size = args['lm']['batch_size']

        


    def forward(self, x, hiddens=None, return_h=False):
        emb = self.encoder(x)
        
        raw_output = emb
        outputs = []
        new_hiddens = []
        if hiddens is None:
            hiddens = self.init_hidden(self.batch_size)
        for i, rnn in enumerate(self.rnns):
            raw_output, new_hidden = rnn(raw_output, hiddens[i])
            outputs.append(raw_output)
            new_hiddens.append(new_hidden)

        # result = raw_output.view(raw_output.size(0)*raw_output.size(1), raw_output.size(2))
        result = self.decoder(raw_output)

        if return_h:
            return result, new_hiddens, outputs
        else:
            return result, new_hiddens

    def init_hidden(self, batch_size):
        return [(torch.zeros(batch_size, 1, self.lstm_io_size), torch.zeros(batch_size, 1, self.lstm_io_size)),
                (torch.zeros(batch_size, 1, self.lstm_hidden_size), torch.zeros(batch_size, 1, self.lstm_hidden_size)),
                (torch.zeros(batch_size, 1, self.lstm_hidden_size), torch.zeros(batch_size, 1, self.lstm_hidden_size))]


if __name__ == "__main__":
    args = {'device': torch.device('cpu')}
    lm = AWD_LSTM(args)
    torch.save(lm.state_dict(), 'awd_lstm_lm.params')
