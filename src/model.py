"""Neural networks for all experiments."""
import snntorch
from snntorch import surrogate
import torch


class LIF(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, num_output):
        super().__init__()
        self.timesteps = timesteps
        self.hidden = hidden
        self.num_output = num_output

        spike_grad = surrogate.atan()
        reset_mechanism = "subtract"

        beta_out1 = torch.rand(self.hidden)
        thr_out1 = torch.rand(self.hidden)
        self.fc1 = torch.nn.Linear(1, out_features=self.hidden)
        self.lif1 = snntorch.Leaky(
            beta=beta_out1,
            threshold=thr_out1,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        beta_out2 = torch.rand(self.hidden)
        thr_out2 = torch.rand(self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif2 = snntorch.Leaky(
            beta=beta_out2,
            threshold=thr_out2,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        beta_out3 = torch.rand(self.hidden)
        thr_out3 = torch.rand(self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif3 = snntorch.Leaky(
            beta=beta_out3,
            threshold=thr_out3,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        beta_out4 = torch.rand(self.hidden)
        thr_out4 = torch.rand(self.hidden)
        self.fc4 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif4 = snntorch.Leaky(
            beta=beta_out4,
            threshold=thr_out4,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

        beta_out5 = torch.rand(self.num_output)
        thr_out5 = torch.rand(self.num_output)
        self.fc5 = torch.nn.Linear(self.hidden, out_features=self.num_output)
        self.lif5 = snntorch.Leaky(
            beta=beta_out5,
            threshold=thr_out5,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

        # DEBUG
        # alpha_d1 = torch.rand(self.num_output)
        # beta_d1 = torch.rand(self.num_output)
        # thr_d1 = torch.rand(self.num_output)
        # self.fcd1 = torch.nn.Linear(1, self.num_output)
        # self.syn1 = snntorch.Synaptic(
        #     alpha=alpha_d1,
        #     beta=beta_d1,
        #     threshold=thr_d1,
        #     learn_alpha=True,
        #     learn_beta=True,
        #     learn_threshold=True,
        #     spike_grad=spike_grad,
        #     reset_mechanism=reset_mechanism,
        # )
        # alpha_d2 = torch.rand(self.num_output)
        # beta_d2 = torch.rand(self.num_output)
        # thr_d2 = torch.rand(self.num_output)
        # self.fcd2 = torch.nn.Linear(self.hidden, self.num_output)
        # self.syn2 = snntorch.Synaptic(
        #     alpha=alpha_d2,
        #     beta=beta_d2,
        #     threshold=thr_d2,
        #     learn_alpha=True,
        #     learn_beta=True,
        #     learn_threshold=True,
        #     spike_grad=spike_grad,
        #     reset_mechanism=reset_mechanism,
        # )

        params = sum(param.numel() for param in self.parameters())
        space = 20
        print(
            f"{79 * '='}\n"
            f"{' ':<20}{'LIF':^39}{' ':>20}\n"
            f"{79 * '-'}\n"
            f"{'Snntorch:':<{space}}{snntorch.__version__}\n"
            f"{'Timesteps:':<{space}}{self.timesteps}\n"
            f"{'Parameters:':<{space}}{params}\n"
            f"{'Topology:':<{space}}\n{self}\n"
            f"{79 * '='}"
        )

    def forward(self, x):
        """Forward pass for several time steps."""

        mem_out1 = self.lif1.init_leaky()
        mem_out2 = self.lif2.init_leaky()
        mem_out3 = self.lif3.init_leaky()
        mem_out4 = self.lif4.init_leaky()
        mem_out5 = self.lif5.init_leaky()

        # synd1, memd1 = self.syn1.init_synaptic()
        # synd2, memd2 = self.syn2.init_synaptic()

        cur_out_rec = []
        mem_out_rec = []
        spk_out_rec = []
        spk_12 = []
        spk_23 = []

        for step in range(self.timesteps):
            x_timestep = x[step, :, :]

            cur_out1 = self.fc1(x_timestep)
            spk_out1, mem_out1 = self.lif1(cur_out1, mem_out1)
            cur_out2 = self.fc2(spk_out1)
            spk_out2, mem_out2 = self.lif2(cur_out2, mem_out2)
            cur_out3 = self.fc3(spk_out2)
            spk_out3, mem_out3 = self.lif3(cur_out3, mem_out3)

            # DEBUG
            spk_12.append(spk_out1)
            spk_23.append(spk_out2)
            ##########################

            cur_out4 = self.fc4(mem_out3)
            spk_out4, mem_out4 = self.lif4(cur_out4, mem_out4)
            cur_out5 = self.fc5(mem_out4)
            spk_out5, mem_out5 = self.lif5(cur_out5, mem_out5)

            mem_output = torch.mean(input=mem_out5, dim=-1)
            mem_output = torch.unsqueeze(input=mem_output, dim=-1)

            cur_out_rec.append(cur_out5)
            mem_out_rec.append(mem_output)
            spk_out_rec.append(spk_out5)

        ######Debug##

        spk_12 = torch.mean(torch.stack(spk_12, dim=0), dim=[0, 1, 2])
        spk_23 = torch.mean(torch.stack(spk_23, dim=0), dim=[0, 1, 2])
        print(f"\nspk12: {spk_12}, spk23: {spk_23}")

        ##########

        return {
            "current": torch.stack(cur_out_rec, dim=0),
            "membrane_potential": torch.stack(mem_out_rec, dim=0),
            "spikes": torch.stack(spk_out_rec, dim=0),
            "spk_12": spk_12,
            "spk_23": spk_23,
        }


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

        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.hidden,
            num_layers=3,
        )

        self.fc1 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.act = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(self.hidden, out_features=self.num_output)

        params = sum(param.numel() for param in self.parameters())
        space = 20
        print(
            f"{79 * '='}\n"
            f"{' ':<20}{'LSTM':^39}{' ':>20}\n"
            f"{79 * '-'}\n"
            f"{'Torch:':<{space}}{torch.__version__}\n"
            f"{'Timesteps:':<{space}}{self.timesteps}\n"
            f"{'Parameters:':<{space}}{params}\n"
            f"{'Topology:':<{space}}\n{self}\n"
            f"{79 * '='}"
        )

    def forward(self, x):
        """Forward pass for several time steps."""
        out = self.lstm(x)
        out = self.fc1(out[0])
        out = self.act(out)
        out = self.fc2(out)
        out = torch.mean(input=out, dim=-1)
        out = torch.unsqueeze(input=out, dim=-1)

        return {"membrane_potential": out}


def main():
    """Main function for model.py"""
    lif = LIF(timesteps=10, hidden=256, num_output=64)
    slstm = SLSTM(timesteps=10, hidden=256, num_output=64)
    lstm = LSTM(timesteps=10, hidden=341, num_output=64)
    print(lif, slstm, lstm)
    return None


if __name__ == "__main__":
    main()
