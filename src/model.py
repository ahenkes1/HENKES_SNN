"""Neural networks for all experiments."""
import snntorch
from snntorch import surrogate
import torch


class SLSTM(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, num_output):
        super().__init__()
        self.timesteps = timesteps
        self.hidden = hidden
        self.num_output = num_output

        spike_grad = surrogate.atan()

        thr_lstm_1 = torch.rand(self.hidden)
        self.slstm_1 = snntorch.SLSTM(
            input_size=1,
            hidden_size=self.hidden,
            spike_grad=spike_grad,
            learn_threshold=True,
            threshold=thr_lstm_1,
            reset_mechanism="subtract",
        )

        thr_lstm_2 = torch.rand(self.hidden)
        self.slstm_2 = snntorch.SLSTM(
            input_size=self.hidden,
            hidden_size=self.hidden,
            spike_grad=spike_grad,
            learn_threshold=True,
            threshold=thr_lstm_2,
            reset_mechanism="subtract",
        )

        thr_lstm_3 = torch.rand(self.hidden)
        self.slstm_3 = snntorch.SLSTM(
            input_size=self.hidden,
            hidden_size=self.hidden,
            spike_grad=spike_grad,
            learn_threshold=True,
            threshold=thr_lstm_3,
            reset_mechanism="subtract",
        )

        beta_out = torch.rand(self.hidden)
        thr_out = torch.rand(self.hidden)
        self.fc1 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif1 = snntorch.Leaky(
            beta=beta_out,
            threshold=thr_out,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

        beta_out2 = torch.rand(self.num_output)
        thr_out2 = torch.rand(self.num_output)
        self.fc2 = torch.nn.Linear(self.hidden, out_features=self.num_output)
        self.lif2 = snntorch.Leaky(
            beta=beta_out2,
            threshold=thr_out2,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

        params = sum(param.numel() for param in self.parameters())
        space = 20
        print(
            f"{79 * '='}\n"
            f"{' ':<20}{'SLSTM':^39}{' ':>20}\n"
            f"{79 * '-'}\n"
            f"{'Snntorch:':<{space}}{snntorch.__version__}\n"
            f"{'Timesteps:':<{space}}{self.timesteps}\n"
            f"{'Parameters:':<{space}}{params}\n"
            f"{'Topology:':<{space}}\n{self}\n"
            f"{79 * '='}"
        )

    def forward(self, x):
        """Forward pass for several time steps."""

        syn_lstm_1, mem_lstm_1 = self.slstm_1.init_slstm()
        syn_lstm_2, mem_lstm_2 = self.slstm_2.init_slstm()
        syn_lstm_3, mem_lstm_3 = self.slstm_3.init_slstm()

        mem_out = self.lif1.init_leaky()
        mem_out2 = self.lif2.init_leaky()

        cur_out_rec = []
        mem_out_rec = []
        spk_out_rec = []

        for step in range(self.timesteps):
            x_timestep = x[step, :, :]

            spk_lstm_1, syn_lstm_1, mem_lstm_1 = self.slstm_1(
                x_timestep, syn_lstm_1, mem_lstm_1
            )
            spk_lstm_2, syn_lstm_2, mem_lstm_2 = self.slstm_2(
                spk_lstm_1, syn_lstm_2, mem_lstm_2
            )
            spk_lstm_3, syn_lstm_3, mem_lstm_3 = self.slstm_3(
                spk_lstm_2, syn_lstm_3, mem_lstm_3
            )

            cur_out = self.fc1(mem_lstm_3)
            spk_out, mem_out = self.lif1(cur_out, mem_out)
            cur_out2 = self.fc2(mem_out)
            spk_out2, mem_out2 = self.lif2(cur_out2, mem_out2)

            mem_output = torch.mean(input=mem_out2, dim=-1)
            mem_output = torch.unsqueeze(input=mem_output, dim=-1)

            cur_out_rec.append(cur_out2)
            mem_out_rec.append(mem_output)
            spk_out_rec.append(spk_lstm_3)

        return {
            "current": torch.stack(cur_out_rec, dim=0),
            "membrane_potential": torch.stack(mem_out_rec, dim=0),
            "spikes": torch.stack(spk_out_rec, dim=0),
        }


class LSTM(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, num_output):
        super().__init__()
        self.timesteps = timesteps
        self.hidden = hidden
        self.num_output = num_output

        self.lstm_1 = torch.nn.LSTMCell(
            input_size=1,
            hidden_size=self.hidden,
        )

        self.lstm_2 = torch.nn.LSTMCell(
            input_size=self.hidden,
            hidden_size=self.hidden,
        )

        self.lstm_3 = torch.nn.LSTMCell(
            input_size=self.hidden,
            hidden_size=self.hidden,
        )

        self.fc1 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.a_1 = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(self.hidden, out_features=self.num_output)

        params = sum(param.numel() for param in self.parameters())
        space = 20
        print(
            f"{79 * '='}\n"
            f"{' ':<20}{'SLSTM':^39}{' ':>20}\n"
            f"{79 * '-'}\n"
            f"{'Torch:':<{space}}{torch.__version__}\n"
            f"{'Timesteps:':<{space}}{self.timesteps}\n"
            f"{'Parameters:':<{space}}{params}\n"
            f"{'Topology:':<{space}}\n{self}\n"
            f"{79 * '='}"
        )

    def forward(self, x):
        """Forward pass for several time steps."""

        TODO
        """
        rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
        input = torch.randn(2, 3, 10) # (time_steps, batch, input_size)
        hx = torch.randn(3, 20) # (batch, hidden_size)
        cx = torch.randn(3, 20)
        output = []
        for i in range(input.size()[0]):
        output = torch.stack(output, dim=0)
        """

        syn_lstm_1, mem_lstm_1 = self.slstm_1.init_slstm()
        syn_lstm_2, mem_lstm_2 = self.slstm_2.init_slstm()
        syn_lstm_3, mem_lstm_3 = self.slstm_3.init_slstm()

        mem_out = self.lif1.init_leaky()
        mem_out2 = self.lif2.init_leaky()

        cur_out_rec = []
        mem_out_rec = []
        spk_out_rec = []

        for step in range(self.timesteps):
            x_timestep = x[step, :, :]

            spk_lstm_1, syn_lstm_1, mem_lstm_1 = self.slstm_1(
                x_timestep, syn_lstm_1, mem_lstm_1
            )
            spk_lstm_2, syn_lstm_2, mem_lstm_2 = self.slstm_2(
                spk_lstm_1, syn_lstm_2, mem_lstm_2
            )
            spk_lstm_3, syn_lstm_3, mem_lstm_3 = self.slstm_3(
                spk_lstm_2, syn_lstm_3, mem_lstm_3
            )

            cur_out = self.fc1(mem_lstm_3)
            spk_out, mem_out = self.lif1(cur_out, mem_out)
            cur_out2 = self.fc2(mem_out)
            spk_out2, mem_out2 = self.lif2(cur_out2, mem_out2)

            mem_output = torch.mean(input=mem_out2, dim=-1)
            mem_output = torch.unsqueeze(input=mem_output, dim=-1)

            cur_out_rec.append(cur_out2)
            mem_out_rec.append(mem_output)
            spk_out_rec.append(spk_lstm_3)

        return {
            "current": torch.stack(cur_out_rec, dim=0),
            "membrane_potential": torch.stack(mem_out_rec, dim=0),
            "spikes": torch.stack(spk_out_rec, dim=0),
        }


def main():
    """Main function for model.py"""
    return None


if __name__ == "__main__":
    main()
