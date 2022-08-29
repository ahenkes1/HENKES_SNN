"""Datasets for all experiments"""
import torch
import torch.utils.data
import tqdm


def isotropic_hardening(
    strains,
    yield_stress=torch.as_tensor(300),
    elastic_modulus=torch.as_tensor(2.1e5),
    hardening_modulus=torch.as_tensor(2.1e5 / 100),
):
    """1D isotropic hardening plasticity."""
    yield_stress = torch.as_tensor(yield_stress)
    elastic_modulus = torch.as_tensor(elastic_modulus)
    hardening_modulus = torch.as_tensor(hardening_modulus)

    strain_plastic = torch.zeros(1)
    strain_plastic_equivalent = torch.zeros(1)

    stresses = []
    strains_plastic_equivalent = []
    strains_plastic = []

    def sign(x):
        """sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)"""
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    for index, strain in enumerate(torch.squeeze(strains)):
        stress_trial = elastic_modulus * (strain - strain_plastic)
        yield_limit = (
            yield_stress + hardening_modulus * strain_plastic_equivalent
        )
        yield_function = abs(stress_trial) - yield_limit

        if yield_function < 0:
            stresses.append(stress_trial)
        else:
            plastic_rate_increment = yield_function / (
                hardening_modulus + elastic_modulus
            )

            stresses.append(
                (
                    1
                    - plastic_rate_increment
                    * elastic_modulus
                    / abs(stress_trial)
                )
                * stress_trial
            )

            strain_plastic_equivalent += plastic_rate_increment
            strain_plastic += sign(stress_trial) * plastic_rate_increment

        strains_plastic_equivalent.append(strain_plastic_equivalent)
        strains_plastic.append(strain_plastic)

    stresses = torch.stack(stresses, dim=0)
    strains_plastic_equivalent = torch.stack(strains_plastic_equivalent, dim=0)
    strains_plastic = torch.stack(strains_plastic, dim=0)

    return {
        "stresses": stresses,
        "strains_plastic_equivalent": strains_plastic_equivalent,
        "strains_plastic": strains_plastic,
    }


class Regression_dataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(
        self,
        timesteps,
        num_samples,
        mode,
        yield_stress=300,
        elastic_modulus=2.1e5,
        hardening_modulus=2.1e5 / 100,
    ):
        self.num_samples = num_samples
        strain_lst = []
        stress_lst = []

        if mode == "plasticity":

            with tqdm.trange(num_samples) as pbar:
                for _ in pbar:
                    MAX_STRAIN = float(torch.rand(1)) * 1e-2

                    strain_load = torch.linspace(
                        start=0.0, end=MAX_STRAIN, steps=timesteps // 2
                    )

                    strain_unload = torch.flip(strain_load, [-1])
                    strain = torch.concat([strain_load, strain_unload], -1)
                    strain = torch.unsqueeze(strain, -1)

                    output_dictionary = isotropic_hardening(
                        strains=strain,
                        yield_stress=yield_stress,
                        elastic_modulus=elastic_modulus,
                        hardening_modulus=hardening_modulus,
                    )

                    stress = torch.Tensor(output_dictionary["stresses"])

                    strain_lst.append(strain)
                    stress_lst.append(stress)

                strain_lst = torch.stack(strain_lst, dim=1)
                stress_lst = torch.stack(stress_lst, dim=1)

                std_strain, mean_strain = torch.std_mean(strain_lst)
                std_stress, mean_stress = torch.std_mean(stress_lst)

                strain_norm = (strain_lst - mean_strain) / std_strain
                stress_norm = (stress_lst - mean_stress) / std_stress

                self.features = strain_norm
                self.labels = stress_norm

        else:
            raise NotImplementedError()

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        return self.features[:, idx, :], self.labels[:, idx, :]


def plasticity(
    yield_stress=300,
    elastic_modulus=2.1e5,
    hardening_modulus=2.1e5 / 100,
    batch_size=1,
    num_samples=1,
    timesteps=10,
):
    """Create dataset and dataloader for the plasticity case."""
    print(f"{79 * '='}\n" f"{' ':<20}{'Dataset':^39}{' ':>20}")
    dataset = Regression_dataset(
        timesteps=timesteps,
        num_samples=num_samples,
        mode="plasticity",
        yield_stress=yield_stress,
        elastic_modulus=elastic_modulus,
        hardening_modulus=hardening_modulus,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    space = 20
    print(
        f"{79 * '-'}\n"
        f"{'samples:':<{space}}{num_samples}\n"
        f"{'timesteps:':<{space}}{timesteps}\n"
        f"{'Youngs modulus:':<{space}}{elastic_modulus}\n"
        f"{'Hardening modulus:':<{space}}{hardening_modulus}\n"
        f"{'Yield stress:':<{space}}{yield_stress}\n"
        f"{'batch size:':<{space}}{batch_size}\n"
        f"{'shuffle:':<{space}}{True}\n"
        f"{79 * '='}"
    )

    return dataloader


def main():
    """Main script for dataset module."""
    dataloader = plasticity(
        yield_stress=300,
        elastic_modulus=2.1e5,
        hardening_modulus=2.1e5 / 100,
        batch_size=1,
        num_samples=1,
        timesteps=10,
    )
    print(next(iter(dataloader)))
    return None


if __name__ == "__main__":
    main()
