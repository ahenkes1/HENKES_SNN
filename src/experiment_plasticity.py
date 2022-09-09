"""Plasticity using SNN. Section ...."""
import csv
import dataset
import model
import matplotlib.pyplot as plt
import trainer
import torch


def convergence(
    device,
    data_train,
    data_val,
    data_test,
    timesteps,
    epochs,
):
    """Convergence study hidden versus num outputs."""
    with open(file=r"./saved_model/results_convergence.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(["out", "hidden", "mean_rel", "mean_rel_end"])

    for out in [16, 32, 64, 128, 256]:
        for hidden in [16, 32, 64, 128, 256]:
            savepath = (
                "./saved_model/"
                + "out_"
                + str(out)
                + "_hidden_"
                + str(hidden)
                + "_"
            )

            slstm = model.SLSTM(
                timesteps=timesteps, hidden=hidden, num_output=out
            ).to(device=device)

            training_hist = trainer.training(
                dataloader_train=data_train,
                dataloader_val=data_val,
                model=slstm,
                learning_rate=1e-3,
                optimizer="adamw",
                device=device,
                epochs=epochs,
                savepath=(savepath + "saved_model.pth"),
            )

            slstm.load_state_dict(torch.load(savepath + "saved_model.pth"))

            testing_results = trainer.test(
                dataloader_test=data_test,
                model=slstm,
                device=device,
            )

            # prediction = trainer.predict(
            #     dataloader_test=data_test,
            #     model=slstm,
            #     device=device,
            #     mean_strain=mean_strain,
            #     std_strain=std_strain,
            #     mean_stress=mean_stress,
            #     std_stress=std_stress,
            #     num_samples=PREDICTION_SIZE,
            # )

            torch.save(
                {
                    "training_hist": training_hist,
                    "testing_results": testing_results,
                    # "prediction": prediction,
                },
                savepath + "save",
            )

            entry = [
                str(out),
                str(hidden),
                testing_results["mean_rel_err_test"],
                testing_results["mean_rel_err_end_test"],
            ]
            with open(
                file=r"./saved_model/results_convergence.csv", mode="a"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(entry)

            # plt.figure(0)
            # for i in range(PREDICTION_SIZE):
            #     plt.plot(
            #         prediction["strain"][:, i],
            #         prediction["true"][:, i],
            #         color="black",
            #     )
            #     plt.plot(
            #         prediction["strain"][:, i],
            #         prediction["prediction"][:, i],
            #         color="red",
            #     )
            # plt.savefig(savepath + "prediction.png")

            plt.figure(1)
            plt.semilogy(training_hist["epoch_loss_train"], label="train")
            plt.semilogy(training_hist["epoch_loss_val"], label="val")
            plt.savefig(savepath + "loss.png")

    return None


def comparison(
    device,
    data_train,
    data_val,
    data_test,
    data_predict,
    timesteps,
    epochs,
    mean_strain,
    std_strain,
    mean_stress,
    std_stress,
):
    """Comparison SLSTM versus LSTM."""
    strain = None
    stress = None
    slstm_stress = None
    lstm_stress = None
    slstm_test = None
    slstm_test_end = None
    lstm_test = None
    lstm_test_end = None

    for case in ["slstm", "lstm"]:

        if case == "slstm":
            savepath = "./saved_model/slstm_"
            network = model.SLSTM(
                timesteps=timesteps, hidden=256, num_output=64
            ).to(device=device)

        else:
            savepath = "./saved_model/lstm_"
            network = model.LSTM(
                timesteps=timesteps, hidden=256, num_output=64
            ).to(device=device)

        training_hist = trainer.training(
            dataloader_train=data_train,
            dataloader_val=data_val,
            model=network,
            learning_rate=1e-3,
            optimizer="adamw",
            device=device,
            epochs=epochs,
            savepath=(savepath + "saved_model.pth"),
        )

        network.load_state_dict(torch.load(savepath + "saved_model.pth"))

        testing_results = trainer.test(
            dataloader_test=data_test,
            model=network,
            device=device,
        )

        prediction = trainer.predict(
            dataloader_predict=data_predict,
            model=network,
            device=device,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            num_samples=1,
        )

        torch.save(
            {
                "training_hist": training_hist,
                "testing_results": testing_results,
                "prediction": prediction,
            },
            savepath + "save",
        )

        if case == "slstm":
            slstm_test = testing_results["mean_rel_err_test"]
            slstm_test_end = testing_results["mean_rel_err_end_test"]
            strain = prediction["strain"].tolist()
            stress = prediction["true"].tolist()
            slstm_stress = prediction["prediction"].tolist()

        else:
            lstm_test = testing_results["mean_rel_err_test"]
            lstm_test_end = testing_results["mean_rel_err_end_test"]
            lstm_stress = prediction["prediction"].tolist()

    # strain = reduce(lambda x, y: x + y, strain)
    # stress = reduce(lambda x, y: x + y, stress)

    strain_stress = list(zip(strain, stress))
    strain_slstm = list(zip(strain, slstm_stress))
    strain_lstm = list(zip(strain, lstm_stress))

    with open(file=r"./saved_model/results_comparison.csv", mode="a") as file:
        writer = csv.writer(file)
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
                "Strain-SLSTM",
            ]
        )
        writer.writerow(
            [
                strain_slstm,
            ]
        )
        writer.writerow(
            [
                "Strain-LSTM",
            ]
        )
        writer.writerow(
            [
                strain_lstm,
            ]
        )
        writer.writerow(
            [
                "SLSTM_mean_rel",
                "SLSTM_mean_rel_end",
                "LSTM_mean_rel",
                "LSTM_mean_rel_end",
            ]
        )
        writer.writerow(
            [
                slstm_test,
                slstm_test_end,
                lstm_test,
                lstm_test_end,
            ]
        )

    return None


def main(device):
    """Main function for the plasticity experiment."""
    YIELD_STRESS = 300
    ELASTIC_MODULUS = 2.1e5
    HARDENING_MODULUS = 2.1e5 / 100
    BATCH_SIZE = 1024
    TIMESTEPS = 100
    # NUM_SAMPLES_TRAIN = BATCH_SIZE * 10
    NUM_SAMPLES_TRAIN = BATCH_SIZE * 1
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    # EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 500
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 1

    data_train_dict = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_TRAIN,
        timesteps=TIMESTEPS,
        mean_strain=None,
        std_strain=None,
        mean_stress=None,
        std_stress=None,
    )
    data_train = data_train_dict["dataloader"]

    statistics = data_train_dict["statistics"]
    mean_strain = statistics["mean_strain"]
    std_strain = statistics["std_strain"]
    mean_stress = statistics["mean_stress"]
    std_stress = statistics["std_stress"]

    data_val = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_VAL,
        timesteps=TIMESTEPS,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )["dataloader"]
    data_test = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_TEST,
        timesteps=TIMESTEPS,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )["dataloader"]
    data_predict = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=1,
        num_samples=1,
        timesteps=TIMESTEPS,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )["dataloader"]

    # convergence(
    #    device=device,
    #    data_train=data_train,
    #    data_val=data_val,
    #    data_test=data_test,
    #    timesteps=TIMESTEPS,
    #    epochs=EPOCHS,
    # )

    comparison(
        device=device,
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        data_predict=data_predict,
        timesteps=TIMESTEPS,
        epochs=EPOCHS,
        mean_strain=mean_strain,
        std_strain=std_strain,
        mean_stress=mean_stress,
        std_stress=std_stress,
    )

    return None


if __name__ == "__main__":
    main(device="cuda")
