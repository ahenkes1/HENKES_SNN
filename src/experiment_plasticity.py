"""Plasticity using SNN. Section ...."""
import csv
import dataset
import model
import matplotlib.pyplot as plt
import trainer
import torch


def main(device):
    """Main function for the plasticity experiment."""
    YIELD_STRESS = 300
    ELASTIC_MODULUS = 2.1e5
    HARDENING_MODULUS = 2.1e5 / 100
    BATCH_SIZE = 1024
    PREDICTION_SIZE = 10
    TIMESTEPS = 100
    NUM_SAMPLES_TRAIN = BATCH_SIZE * 10
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    # NUM_HIDDEN = 32
    # NUM_OUTPUT = NUM_HIDDEN
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE  # * 500

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

    with open(file=r"./saved_model/results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(["out", "hidden", "mean_rel", "mean_rel_end"])

    for out in [16, 32, 64, 128, 256]:
        for hidden in [16, 32, 64, 128, 256]:
            savepath = (
                "./saved_model/"
                + "_out_"
                + str(out)
                + "_hidden_"
                + str(hidden)
                + "_"
            )

            slstm = model.SLSTM(
                timesteps=TIMESTEPS, hidden=hidden, num_output=out
            ).to(device=device)

            training_hist = trainer.training(
                dataloader_train=data_train,
                dataloader_val=data_val,
                model=slstm,
                learning_rate=1e-3,
                optimizer="adamw",
                device=device,
                epochs=EPOCHS,
                savepath=(savepath + "saved_model.pth"),
            )

            slstm.load_state_dict(torch.load(savepath + "saved_model.pth"))

            testing_results = trainer.test(
                dataloader_test=data_test,
                model=slstm,
                device=device,
            )

            prediction = trainer.predict(
                dataloader_test=data_test,
                model=slstm,
                device=device,
                mean_strain=mean_strain,
                std_strain=std_strain,
                mean_stress=mean_stress,
                std_stress=std_stress,
                num_samples=PREDICTION_SIZE,
            )

            torch.save(
                {
                    "training_hist": training_hist,
                    "testing_results": testing_results,
                    "prediction": prediction,
                },
                savepath + "save",
            )

            entry = [
                str(out),
                str(hidden),
                testing_results["mean_rel_err_test"],
                testing_results["mean_rel_err_end_test"],
            ]
            with open(file=r"./saved_model/results.csv", mode="a") as file:
                writer = csv.writer(file)
                writer.writerow(entry)

            plt.figure(0)
            for i in range(PREDICTION_SIZE):
                plt.plot(
                    prediction["strain"][:, i],
                    prediction["true"][:, i],
                    color="black",
                )
                plt.plot(
                    prediction["strain"][:, i],
                    prediction["prediction"][:, i],
                    color="red",
                )
            plt.savefig("./saved_model/prediction.png")

            plt.figure(1)
            plt.semilogy(training_hist["epoch_loss_train"], label="train")
            plt.semilogy(training_hist["epoch_loss_val"], label="val")
            plt.savefig("./saved_model/loss.png")

    return None


if __name__ == "__main__":
    main(device="cpu")
