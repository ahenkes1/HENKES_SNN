# import matplotlib.pyplot as plt
import snntorch
import snntorch.surrogate
import torch
import torchvision
import tqdm

import pilotnet_dataset

BATCH_SIZE = 100
EPOCHS = 30

DOES NOT WORK, RESHAPING PROBABLY BULLSHIT, TRY FLATTEN!


class LIF(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, num_input, num_output):
        super().__init__()
        self.timesteps = timesteps
        self.hidden = hidden
        self.num_output = num_output

        spike_grad = snntorch.surrogate.atan()
        reset_mechanism = "subtract"
        learning = True
        beta = 0.9
        thr = 0.0

        self.fc1 = torch.nn.Linear(num_input, out_features=self.hidden)
        self.lif1 = snntorch.Leaky(
            beta=torch.ones(self.hidden) * beta,
            threshold=torch.ones(self.hidden) * thr,
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        self.fc2 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif2 = snntorch.Leaky(
            beta=torch.ones(self.hidden) * beta,
            threshold=torch.ones(self.hidden) * thr,
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        self.fc3 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif3 = snntorch.Leaky(
            beta=torch.ones(self.hidden) * beta,
            threshold=torch.ones(self.hidden) * thr,
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
        )

        self.fc4 = torch.nn.Linear(self.hidden, out_features=self.hidden)
        self.lif4 = snntorch.Leaky(
            beta=torch.ones(self.hidden) * beta,
            threshold=torch.ones(self.hidden) * thr,
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

        self.fc5 = torch.nn.Linear(self.hidden, out_features=self.num_output)
        self.lif5 = snntorch.Leaky(
            beta=torch.ones(self.hidden) * beta,
            threshold=torch.ones(self.hidden) * thr,
            learn_beta=True,
            learn_threshold=True,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

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

        cur_out_rec = []
        mem_out_rec = []
        spk_out_rec = []
        spk_12 = []
        spk_23 = []

        for step in range(self.timesteps):
            # [TIME, BATCH, FEATURES]
            x_timestep = x[step, :, :]

            cur_out1 = self.fc1(x_timestep)
            spk_out1, mem_out1 = self.lif1(cur_out1, mem_out1)
            cur_out2 = self.fc2(spk_out1)
            spk_out2, mem_out2 = self.lif2(cur_out2, mem_out2)
            cur_out3 = self.fc3(spk_out2)
            spk_out3, mem_out3 = self.lif3(cur_out3, mem_out3)

            spk_12.append(spk_out1)
            spk_23.append(spk_out2)

            cur_out4 = self.fc4(mem_out3)
            spk_out4, mem_out4 = self.lif4(cur_out4, mem_out4)
            cur_out5 = self.fc5(mem_out4)
            spk_out5, mem_out5 = self.lif5(cur_out5, mem_out5)

            mem_output = torch.mean(input=mem_out5, dim=-1)
            mem_output = torch.unsqueeze(input=mem_output, dim=-1)

            cur_out_rec.append(cur_out5)
            mem_out_rec.append(mem_output)
            spk_out_rec.append(spk_out5)

        spk_23 = torch.mean(torch.stack(spk_23, dim=0), dim=[0, 1, 2])

        return {
            "current": torch.stack(cur_out_rec, dim=0),
            "membrane_potential": torch.stack(mem_out_rec, dim=0),
            "spikes": torch.stack(spk_out_rec, dim=0),
            "spk_23": spk_23,
        }


def main():
    # Datasets
    training_set = pilotnet_dataset.PilotNetDataset(
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([33, 100]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        ),
    )
    testing_set = pilotnet_dataset.PilotNetDataset(
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([33, 100]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=20,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=testing_set,
        batch_size=1,
        shuffle=True,
        num_workers=20,
        drop_last=False,
    )

    # feature: torch.Size([1, 3, 33, 100, 16])
    # [BATCH, CHANNEL, X, Y, TIME]

    # label: torch.Size([1, 16])
    # [BATCH, TIME]

    sample = next(iter(train_loader))
    # [TIME, BATCH, FEATURES]
    x = sample[0].reshape((BATCH_SIZE, -1, 16)).swapaxes(0, -1).swapaxes(1, -1)
    y = sample[1].swapaxes(0, -1).unsqueeze(-1)

    # plt.figure(0)
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     fig = sample[0][0, :, :, :, i].swapaxes(0, 2).swapaxes(0, 1)
    #     plt.imshow(fig)
    #     plt.title(str(sample[1][0, i]))

    # plt.tight_layout()
    # plt.show()
    model = LIF(
        timesteps=x.shape[0],
        hidden=256,
        num_input=x.shape[-1],
        num_output=y.shape[-1],
    )
    optimizer = torch.optim.Rprop(model.parameters(), lr=3e-4)
    loss_function = torch.nn.MSELoss()

    loss_mse_train_lst = []
    loss_mse_val_lst = []

    with tqdm.trange(int(EPOCHS)) as pbar:
        for _ in pbar:
            minibatch_counter_train = 0

            train_batch = iter(train_loader)
            for feature, label in train_batch:
                feature = (
                    feature.reshape((BATCH_SIZE, -1, 16))
                    .swapaxes(0, -1)
                    .swapaxes(1, -1)
                )
                label = label.swapaxes(0, -1).unsqueeze(-1)
                mem = model(feature)["membrane_potential"]

                loss_train = loss_function(mem, label)
                loss_mse_train_lst.append(loss_train.item())
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                minibatch_counter_train += 1

            minibatch_counter_val = 0

            val_batch = iter(val_loader)
            for feature, label in val_batch:
                feature = (
                    feature.reshape((1, -1, 16))
                    .swapaxes(0, -1)
                    .swapaxes(1, -1)
                )
                label = label.swapaxes(0, -1).unsqueeze(-1)
                mem = model(feature)["membrane_potential"]

                loss_val = loss_function(mem, label)
                loss_mse_val_lst.append(loss_val.item())

                minibatch_counter_val += 1

            avg_batch_loss_train = (
                sum(loss_mse_train_lst) / minibatch_counter_train
            )
            avg_batch_loss_val = sum(loss_mse_val_lst) / minibatch_counter_val

            pbar.set_postfix(
                loss_train=avg_batch_loss_train,
                loss_val=avg_batch_loss_val,
            )

    return None


if __name__ == "__main__":
    main()
