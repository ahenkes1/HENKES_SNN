"""Plasticity using SNN. Section ...."""
import csv
import dataset
import model
import matplotlib.pyplot as plt
import trainer
import torch


def check(
    timesteps,
    device,
    batch_size,
    num_samples_train,
    num_samples_val,
    num_samples_test,
    hidden,
    epochs,
):
    """Re-check test results."""
    for timesteps in [timesteps]:
        savepath = "./saved_model/" "RO_" + "timesteps_" + str(timesteps) + "_"

        data_train_dict = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_train,
            timesteps=timesteps,
            mean_strain=None,
            std_strain=None,
            mean_stress=None,
            std_stress=None,
            mean_yield=None,
            std_yield=None,
        )
        data_train = data_train_dict["dataloader"]

        statistics = data_train_dict["statistics"]
        mean_strain = statistics["mean_strain"]
        std_strain = statistics["std_strain"]
        mean_stress = statistics["mean_stress"]
        std_stress = statistics["std_stress"]
        mean_yield = statistics["mean_yield"]
        std_yield = statistics["std_yield"]

        data_val = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_val,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]
        data_test = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_test,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]
        data_predict = dataset.ramberg_osgood(
            batch_size=5,
            num_samples=5,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]

        rlif = model.RLIF(
            timesteps=timesteps, hidden=hidden, num_output=hidden
        ).to(device=device)

        rlif.load_state_dict(torch.load(savepath + "saved_model.pth"))

        testing_results = trainer.test(
            dataloader_test=data_test,
            model=rlif,
            device=device,
        )

        plt.figure(0)
        hist_all = testing_results['histo_all'][0]
        hist_end = testing_results['histo_end'][0]

        hist_all_counts, hist_all_bins = np.histogram(hist_all)
        hist_end_counts, hist_end_bins = np.histogram(hist_end)
        
        hist_all_counts = hist_all_counts.tolist()
        hist_all_bins = hist_all_bins.tolist()

        # print(list(zip(hist_all_bins, hist_all_counts)))
        print(hist_all_bins)
        print(hist_all_counts)

        plt.show()
        plt.close(1)
        raise SystemExit()

        plt.figure(1)
        pred_true = testing_results["pred_true"][0]
        plt.scatter(pred_true[:, 0], pred_true[:, 1])
        plt.show()
        plt.close(1)

        prediction = trainer.predict(
            dataloader_predict=data_predict,
            model=rlif,
            device=device,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=mean_yield,
            num_samples=5,
        )

        plt.figure(2)
        for i in range(5):
            plt.plot(
                prediction["strain"][:, i],
                prediction["true"][:, i],
                label="True",
            )
            plt.plot(
                prediction["strain"][:, i],
                prediction["prediction"][:, i],
                label="LIF",
            )
            plt.legend()
        # plt.savefig(savepath + "prediction.png")
        plt.show()

        # torch.save(
        #     {
        #         "training_hist": training_hist,
        #         "testing_results": testing_results,
        #         "prediction": prediction,
        #     },
        #     savepath + "save",
        #     )
        plt.close(2)

        # for i in range(5):
        #     strain = prediction["strain"][:, i].tolist()
        #     stress = prediction["true"][:, i].tolist()
        #     slstm_stress = prediction["prediction"][:, i].tolist()
        #     strain_stress = list(zip(strain, stress))
        #     strain_rlif = list(zip(strain, slstm_stress))

        #     with open(
        #         file=r"./saved_model/results_convergence_ramberg.csv", mode="a"
        #     ) as file:
        #         writer = csv.writer(file)
        #         writer.writerow(
        #             [
        #                 "timesteps",
        #                 "mean_rel_err_test",
        #                 "mean_rel_err_end_test",
        #             ]
        #         )
        #         writer.writerow(
        #             [
        #                 str(timesteps),
        #                 testing_results["mean_rel_err_test"],
        #                 testing_results["mean_rel_err_end_test"],
        #             ]
        #         )
        #         writer.writerow(
        #             [
        #                 "Strain-Stress",
        #             ]
        #         )
        #         writer.writerow(
        #             [
        #                 strain_stress,
        #             ]
        #         )
        #         writer.writerow(
        #             [
        #                 "Strain-RLIF",
        #             ]
        #         )
        #         writer.writerow(
        #             [
        #                 strain_rlif,
        #             ]
        #         )
        #         writer.writerow([])

    return None


def convergence(
    timesteps,
    device,
    batch_size,
    num_samples_train,
    num_samples_val,
    num_samples_test,
    hidden,
    epochs,
):
    """Convergence study hidden versus num outputs."""
    for timesteps in [timesteps]:
        savepath = "./saved_model/" "RO_" + "timesteps_" + str(timesteps) + "_"

        data_train_dict = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_train,
            timesteps=timesteps,
            mean_strain=None,
            std_strain=None,
            mean_stress=None,
            std_stress=None,
            mean_yield=None,
            std_yield=None,
        )
        data_train = data_train_dict["dataloader"]

        statistics = data_train_dict["statistics"]
        mean_strain = statistics["mean_strain"]
        std_strain = statistics["std_strain"]
        mean_stress = statistics["mean_stress"]
        std_stress = statistics["std_stress"]
        mean_yield = statistics["mean_yield"]
        std_yield = statistics["std_yield"]

        data_val = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_val,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]
        data_test = dataset.ramberg_osgood(
            batch_size=batch_size,
            num_samples=num_samples_test,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]
        data_predict = dataset.ramberg_osgood(
            batch_size=5,
            num_samples=5,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=std_yield,
        )["dataloader"]

        rlif = model.RLIF(
            timesteps=timesteps, hidden=hidden, num_output=hidden
        ).to(device=device)

        training_hist = trainer.training(
            dataloader_train=data_train,
            dataloader_val=data_val,
            model=rlif,
            learning_rate=1e-3,
            optimizer="adamw",
            device=device,
            epochs=epochs,
            savepath=(savepath + "saved_model.pth"),
        )

        rlif.load_state_dict(torch.load(savepath + "saved_model.pth"))

        testing_results = trainer.test(
            dataloader_test=data_test,
            model=rlif,
            device=device,
        )

        torch.save(
            {
                "training_hist": training_hist,
                "testing_results": testing_results,
            },
            savepath + "save",
        )

        plt.figure(1)
        plt.semilogy(training_hist["epoch_loss_train"], label="train")
        plt.semilogy(training_hist["epoch_loss_val"], label="val")
        plt.savefig(savepath + "loss.png")

        prediction = trainer.predict(
            dataloader_predict=data_predict,
            model=rlif,
            device=device,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            mean_yield=mean_yield,
            std_yield=mean_yield,
            num_samples=5,
        )
        plt.close(1)

        plt.figure(2)
        for i in range(5):
            plt.plot(
                prediction["strain"][:, i],
                prediction["true"][:, i],
                label="True",
            )
            plt.plot(
                prediction["strain"][:, i],
                prediction["prediction"][:, i],
                label="LIF",
            )
            plt.legend()
        plt.savefig(savepath + "prediction.png")

        torch.save(
            {
                "training_hist": training_hist,
                "testing_results": testing_results,
                "prediction": prediction,
            },
            savepath + "save",
        )
        plt.close(2)

        for i in range(5):
            strain = prediction["strain"][:, i].tolist()
            stress = prediction["true"][:, i].tolist()
            slstm_stress = prediction["prediction"][:, i].tolist()
            strain_stress = list(zip(strain, stress))
            strain_rlif = list(zip(strain, slstm_stress))

            with open(
                file=r"./saved_model/results_convergence_ramberg.csv", mode="a"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timesteps",
                        "mean_rel_err_test",
                        "mean_rel_err_end_test",
                    ]
                )
                writer.writerow(
                    [
                        str(timesteps),
                        testing_results["mean_rel_err_test"],
                        testing_results["mean_rel_err_end_test"],
                    ]
                )
                writer.writerow(
                    [
                        "Strain-Stress",
                    ]
                )
                writer.writerow(
                    [
                        strain_stress,
                    ]
                )
                writer.writerow(
                    [
                        "Strain-RLIF",
                    ]
                )
                writer.writerow(
                    [
                        strain_rlif,
                    ]
                )
                writer.writerow([])

    return None


def main(device):
    """Main function for the Ramberg-Osgood experiment."""
    BATCH_SIZE = 1024
    TIMESTEPS = 20
    NUM_SAMPLES_TRAIN = BATCH_SIZE * 1
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 5000

    convergence(
        timesteps=TIMESTEPS,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_samples_train=NUM_SAMPLES_TRAIN,
        num_samples_val=NUM_SAMPLES_VAL,
        num_samples_test=NUM_SAMPLES_TEST,
        hidden=512,
    )

    return None


if __name__ == "__main__":
    # main(device="cuda")

    import numpy as np
    import random
    import torch

    # Seeds
    SEED = 42
    # Python RNG
    random.seed(SEED)
    # Numpy RNG
    np.random.seed(SEED)
    # TF RNG
    torch.manual_seed(SEED)

    BATCH_SIZE = 1024
    TIMESTEPS = 20
    NUM_SAMPLES_TRAIN = BATCH_SIZE * 1
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 5000
    device = torch.device("cuda")
    check(
        timesteps=TIMESTEPS,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_samples_train=NUM_SAMPLES_TRAIN,
        num_samples_val=NUM_SAMPLES_VAL,
        num_samples_test=NUM_SAMPLES_TEST,
        hidden=512,
    )
