import matplotlib.pyplot as plt
import torch
import torchvision

import pilotnet_dataset

BATCH_SIZE = 1

# Datasets
training_set = pilotnet_dataset.PilotNetDataset(
    train=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([33, 100]),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
testing_set = pilotnet_dataset.PilotNetDataset(
    train=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([33, 100]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

train_loader = torch.utils.data.DataLoader(
    dataset=training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)
test_loader = torch.utils.data.DataLoader(
    dataset=testing_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)

# feature: torch.Size([1, 3, 33, 100, 16])
# [BATCH, CHANNEL, X, Y, TIME]

# label: torch.Size([1, 16])
# [BATCH, TIME]

sample = next(iter(train_loader))

plt.figure(0)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    fig = sample[0][0, :, :, :, i].swapaxes(0, 2).swapaxes(0, 1)
    plt.imshow(fig)
    plt.title(str(sample[1][0, i]))

plt.tight_layout()
plt.show()
