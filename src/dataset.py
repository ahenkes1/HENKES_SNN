"""Datasets for all experiments"""
import torch
import torch.utils.data
import tqdm

# from pprint import pprint


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


def newton_ramberg_osgood(
    strain, youngs_modulus, sigma_0, n, sigma_n, tol, max_iter
):
    """Newton iteration for Ramberg-Osgood."""
    strain = torch.as_tensor(strain)
    youngs_modulus = torch.as_tensor(youngs_modulus)
    sigma_0 = torch.as_tensor(sigma_0)
    n = torch.as_tensor(n)
    sigma_n = torch.as_tensor(sigma_n)
    tol = torch.as_tensor(tol)

    iteration = None
    for iteration in range(max_iter):

        residual = strain - (
            (sigma_n / youngs_modulus) + (0.002 * (sigma_n / sigma_0) ** n)
        )

        if torch.norm(residual) < tol and iteration > 0:
            break

        d_residual = (-1 / youngs_modulus) - (
            ((0.002 * n) / sigma_0) * ((sigma_n / sigma_0) ** (n - 1))
        )

        sigma_n = sigma_n - (residual / d_residual)

    if iteration == max_iter:
        raise SystemExit("Max. iter. reached!")

    return sigma_n


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
        n=None,
        mean_strain=None,
        std_strain=None,
        mean_stress=None,
        std_stress=None,
        mean_youngs=None,
        std_youngs=None,
    ):
        self.num_samples = num_samples
        self.mean_strain = mean_strain
        self.std_strain = std_strain
        self.mean_stress = mean_stress
        self.std_stress = std_stress
        self.mean_youngs = mean_youngs
        self.std_youngs = std_youngs
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

            if mean_strain is None:
                print("Calculate mean and std!")
                std_strain, mean_strain = torch.std_mean(strain_lst)
                std_stress, mean_stress = torch.std_mean(stress_lst)

                self.mean_strain = mean_strain
                self.std_strain = std_strain

                self.mean_stress = mean_stress
                self.std_stress = std_stress

            else:
                print("Using pre-defined mean and std!")

            strain_norm = (strain_lst - self.mean_strain) / self.std_strain
            stress_norm = (stress_lst - self.mean_stress) / self.std_stress

            self.features = strain_norm
            self.labels = stress_norm

        elif mode == "ramberg_osgood":

            with tqdm.trange(num_samples) as pbar:
                for _ in pbar:
                    MAX_STRAIN = float(torch.rand(1)) * 1e-2
                    # ELASTIC_MODULUS = float(torch.rand(1)) * 1e5

                    strain = torch.linspace(
                        start=0.0, end=MAX_STRAIN, steps=timesteps
                    )
                    strain = torch.unsqueeze(strain, -1)

                    stress = newton_ramberg_osgood(
                        strain=strain,
                        youngs_modulus=elastic_modulus,
                        sigma_0=yield_stress,
                        n=n,
                        sigma_n=strain * elastic_modulus,
                        tol=1e-8,
                        max_iter=100,
                    )

                    strain_lst.append(strain)
                    stress_lst.append(stress)

            strain_lst = torch.stack(strain_lst, dim=1)
            stress_lst = torch.stack(stress_lst, dim=1)

            if mean_strain is None:
                print("Calculate mean and std!")
                std_strain, mean_strain = torch.std_mean(strain_lst)
                std_stress, mean_stress = torch.std_mean(stress_lst)

                self.mean_strain = mean_strain
                self.std_strain = std_strain

                self.mean_stress = mean_stress
                self.std_stress = std_stress

        elif mode == "elasticity":

            # modulus_lst = []
            with tqdm.trange(num_samples) as pbar:
                for _ in pbar:
                    MAX_STRAIN = float(torch.rand(1)) * 1e-2
                    # MAX_STRAIN = 1e-2
                    # ELASTIC_MODULUS = float(torch.rand(1)) * 1e5

                    strain = torch.linspace(
                        start=0.0, end=MAX_STRAIN, steps=timesteps
                    )
                    strain = torch.unsqueeze(strain, -1)

                    stress = torch.Tensor(elastic_modulus * strain)

                    # modulus_lst.append(torch.as_tensor(elastic_modulus))
                    strain_lst.append(strain)
                    stress_lst.append(stress)

            # modulus_lst = torch.stack(modulus_lst, dim=0)
            strain_lst = torch.stack(strain_lst, dim=1)
            stress_lst = torch.stack(stress_lst, dim=1)

            # if mean_youngs is None and std_youngs is None:
            #     print("Calculate mean and std!")
            #     std_youngs, mean_youngs = torch.std_mean(modulus_lst)
            #     self.mean_youngs = mean_youngs
            #     self.std_youngs = std_youngs

            # else:
            #     print("Using pre-defined mean and std!")

            if mean_strain is None:
                print("Calculate mean and std!")
                std_strain, mean_strain = torch.std_mean(strain_lst)
                std_stress, mean_stress = torch.std_mean(stress_lst)

                self.mean_strain = mean_strain
                self.std_strain = std_strain

                self.mean_stress = mean_stress
                self.std_stress = std_stress

            else:
                print("Using pre-defined mean and std!")

            # youngs_norm = (modulus_lst - self.mean_youngs) / self.std_youngs
            strain_norm = (strain_lst - self.mean_strain) / self.std_strain
            stress_norm = (stress_lst - self.mean_stress) / self.std_stress

            # youngs_norm = torch.unsqueeze(youngs_norm, 0)
            # youngs_norm = torch.unsqueeze(youngs_norm, -1)
            # youngs_norm = torch.repeat_interleave(youngs_norm, timesteps, 0)
            # self.features = torch.cat([strain_norm, youngs_norm], dim=-1)
            self.features = strain_norm
            self.labels = stress_norm
            # self.youngs = youngs_norm

        else:
            raise NotImplementedError()

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation."""
        return self.features[:, idx, :], self.labels[:, idx, :]

    def statistics(self):
        """Return mean and std."""
        return {
            "mean_strain": self.mean_strain,
            "std_strain": self.std_strain,
            "mean_stress": self.mean_stress,
            "std_stress": self.std_stress,
            "mean_youngs": self.mean_youngs,
            "std_youngs": self.std_youngs,
        }


def plasticity(
    yield_stress=300,
    elastic_modulus=2.1e5,
    hardening_modulus=2.1e5 / 100,
    batch_size=1,
    num_samples=1,
    timesteps=10,
    mean_strain=None,
    std_strain=None,
    mean_stress=None,
    std_stress=None,
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
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )

    statistics = dataset.statistics()

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

    return {"dataloader": dataloader, "statistics": statistics}


def ramberg_osgood(
    yield_stress=None,
    elastic_modulus=None,
    hardening_modulus=None,
    batch_size=1,
    num_samples=1,
    timesteps=10,
    n=None,
    mean_strain=None,
    std_strain=None,
    mean_stress=None,
    std_stress=None,
):
    """Create dataset and dataloader for the Ramberg-Osgood case."""
    print(f"{79 * '='}\n" f"{' ':<20}{'Dataset':^39}{' ':>20}")
    dataset = Regression_dataset(
        timesteps=timesteps,
        num_samples=num_samples,
        mode="ramberg_osgood",
        yield_stress=yield_stress,
        elastic_modulus=elastic_modulus,
        hardening_modulus=hardening_modulus,
        n=n,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )

    statistics = dataset.statistics()

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
        f"{'batch size:':<{space}}{batch_size}\n"
        f"{'shuffle:':<{space}}{True}\n"
        f"{79 * '='}"
    )

    return {"dataloader": dataloader, "statistics": statistics}


def elasticity(
    yield_stress=None,
    elastic_modulus=None,
    hardening_modulus=None,
    batch_size=1,
    num_samples=1,
    timesteps=10,
    mean_strain=None,
    std_strain=None,
    mean_stress=None,
    std_stress=None,
    mean_youngs=None,
    std_youngs=None,
):
    """Create dataset and dataloader for the elasticity case."""
    print(f"{79 * '='}\n" f"{' ':<20}{'Dataset':^39}{' ':>20}")
    dataset = Regression_dataset(
        timesteps=timesteps,
        num_samples=num_samples,
        mode="elasticity",
        yield_stress=yield_stress,
        elastic_modulus=elastic_modulus,
        hardening_modulus=hardening_modulus,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
        mean_youngs=mean_youngs,
        std_youngs=std_youngs,
    )

    statistics = dataset.statistics()

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
        f"{'batch size:':<{space}}{batch_size}\n"
        f"{'shuffle:':<{space}}{True}\n"
        f"{79 * '='}"
    )

    return {"dataloader": dataloader, "statistics": statistics}


def main():
    """Main script for dataset module."""
    import csv
    from functools import reduce

    dataloader = plasticity(
        yield_stress=300,
        elastic_modulus=2.1e5,
        hardening_modulus=2.1e5 / 100,
        batch_size=1,
        num_samples=5,
        timesteps=100,
    )

    with open(file=r"./saved_model/plasticity.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(["strain-stress"])

    statistics = dataloader["statistics"]
    mean_strain, std_strain, mean_stress, std_stress = statistics.values()

    dataloader = iter(dataloader["dataloader"])
    for sample in range(5):
        strain, stress = next(dataloader)

        strain = (strain * std_strain) + mean_strain
        stress = (stress * std_stress) + mean_stress

        strain = strain.cpu().tolist()[0]
        stress = stress.cpu().tolist()[0]

        strain = reduce(lambda x, y: x + y, strain)
        stress = reduce(lambda x, y: x + y, stress)

        entry = list(zip(strain, stress))

        with open(file=r"./saved_model/plasticity.csv", mode="a") as file:
            writer = csv.writer(file)
            writer.writerow(entry)

    return None


if __name__ == "__main__":
    main()
    # strain_lst = []
    # stress_lst = []
    # for strain in torch.linspace(0.0, 0.01, 100):
    #     stress = newton_ramberg_osgood(
    #         strain=strain,
    #         youngs_modulus=2.1e5,
    #         sigma_0=300,
    #         n=5,
    #         sigma_n=strain * 2.1e5,
    #         tol=1e-8,
    #         max_iter=100,
    #     )

    #     strain_lst.append(strain.cpu().tolist())
    #     stress_lst.append(stress.cpu().tolist())

    # strain_stress = list(zip(strain_lst, stress_lst))

    # import matplotlib.pyplot as plt

    # plt.plot(strain_lst, stress_lst)
    # plt.show()
    # import matplotlib.pyplot as plt

    # ns = 1000
    # ela_dict = elasticity(elastic_modulus=1, num_samples=ns, timesteps=10)
    # dataloader = ela_dict["dataloader"]
    # statistics  = ela_dict["statistics"]
    # print(statistics)
    # data = iter(dataloader)
    # for feature, label in data:
    #     feature = torch.squeeze(feature)
    #     label = torch.squeeze(label)
    #     plt.plot(feature, label)

    # plt.show()
