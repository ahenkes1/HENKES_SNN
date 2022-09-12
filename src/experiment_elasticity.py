"""Plasticity using SNN. Section ...."""
import csv
import dataset
import model
import matplotlib.pyplot as plt
import trainer
import torch


def convergence(
    device,
    elastic_modulus,
    batch_size,
    num_samples_train,
    num_samples_val,
    num_samples_test,
    hidden,
    epochs,
):
    """Convergence study hidden versus num outputs."""
    for timesteps in [2, 5, 10, 20, 50, 100]:
        savepath = "./saved_model/" + "timesteps_" + str(timesteps) + "_"

        data_train_dict = dataset.elasticity(
            elastic_modulus=elastic_modulus,
            batch_size=batch_size,
            num_samples=num_samples_train,
            timesteps=timesteps,
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

        data_val = dataset.elasticity(
            elastic_modulus=elastic_modulus,
            batch_size=batch_size,
            num_samples=num_samples_val,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
        )["dataloader"]
        data_test = dataset.elasticity(
            elastic_modulus=elastic_modulus,
            batch_size=batch_size,
            num_samples=num_samples_test,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
        )["dataloader"]
        data_predict = dataset.elasticity(
            elastic_modulus=elastic_modulus,
            batch_size=1,
            num_samples=1,
            timesteps=timesteps,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
        )["dataloader"]

        lif = model.LIF(
            timesteps=timesteps, hidden=hidden, num_output=hidden
        ).to(device=device)

        training_hist = trainer.training(
            dataloader_train=data_train,
            dataloader_val=data_val,
            model=lif,
            learning_rate=1e-3,
            optimizer="adamw",
            device=device,
            epochs=epochs,
            savepath=(savepath + "saved_model.pth"),
        )

        lif.load_state_dict(torch.load(savepath + "saved_model.pth"))

        testing_results = trainer.test(
            dataloader_test=data_test,
            model=lif,
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
            model=lif,
            device=device,
            mean_strain=mean_strain,
            std_strain=std_strain,
            mean_stress=mean_stress,
            std_stress=std_stress,
            num_samples=1,
        )
        plt.close(1)

        plt.figure(2)
        plt.plot(prediction["strain"], prediction["true"], label="True")
        plt.plot(prediction["strain"], prediction["prediction"], label="LIF")
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

        strain = prediction["strain"].tolist()
        stress = prediction["true"].tolist()
        slstm_stress = prediction["prediction"].tolist()
        strain_stress = list(zip(strain, stress))
        strain_lif = list(zip(strain, slstm_stress))

        with open(
            file=r"./saved_model/results_convergence_elasticity.csv", mode="a"
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
                    "Strain-LIF",
                ]
            )
            writer.writerow(
                [
                    strain_lif,
                ]
            )
            writer.writerow([])

    return None


# def comparison(
#     device,
#     data_train,
#     data_val,
#     data_test,
#     data_predict,
#     timesteps,
#     epochs,
#     mean_strain,
#     std_strain,
#     mean_stress,
#     std_stress,
# ):
#     """Comparison SLSTM versus LSTM."""
#     strain = None
#     stress = None
#     slstm_stress = None
#     lstm_stress = None
#     slstm_test = None
#     slstm_test_end = None
#     lstm_test = None
#     lstm_test_end = None
#
#     for case in ["slstm", "lstm"]:
#
#         if case == "slstm":
#             savepath = "./saved_model/slstm_"
#             network = model.SLSTM(
#                 timesteps=timesteps, hidden=256, num_output=64
#             ).to(device=device)
#
#         else:
#             savepath = "./saved_model/lstm_"
#             network = model.LSTM(
#                 timesteps=timesteps, hidden=256, num_output=64
#             ).to(device=device)
#
#         training_hist = trainer.training(
#             dataloader_train=data_train,
#             dataloader_val=data_val,
#             model=network,
#             learning_rate=1e-3,
#             optimizer="adamw",
#             device=device,
#             epochs=epochs,
#             savepath=(savepath + "saved_model.pth"),
#         )
#
#         network.load_state_dict(torch.load(savepath + "saved_model.pth"))
#
#         testing_results = trainer.test(
#             dataloader_test=data_test,
#             model=network,
#             device=device,
#         )
#
#         prediction = trainer.predict(
#             dataloader_predict=data_predict,
#             model=network,
#             device=device,
#             mean_strain=mean_strain,
#             std_strain=std_strain,
#             mean_stress=mean_stress,
#             std_stress=std_stress,
#             num_samples=1,
#         )
#
#         torch.save(
#             {
#                 "training_hist": training_hist,
#                 "testing_results": testing_results,
#                 "prediction": prediction,
#             },
#             savepath + "save",
#         )
#
#         if case == "slstm":
#             slstm_test = testing_results["mean_rel_err_test"]
#             slstm_test_end = testing_results["mean_rel_err_end_test"]
#             strain = prediction["strain"].tolist()
#             stress = prediction["true"].tolist()
#             slstm_stress = prediction["prediction"].tolist()
#
#         else:
#             lstm_test = testing_results["mean_rel_err_test"]
#             lstm_test_end = testing_results["mean_rel_err_end_test"]
#             lstm_stress = prediction["prediction"].tolist()
#
#     strain_stress = list(zip(strain, stress))
#     strain_slstm = list(zip(strain, slstm_stress))
#     strain_lstm = list(zip(strain, lstm_stress))
#
#     with open(file=r"./saved_model/results_comparison.csv", mode="a") as file:
#         writer = csv.writer(file)
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
#                 "Strain-SLSTM",
#             ]
#         )
#         writer.writerow(
#             [
#                 strain_slstm,
#             ]
#         )
#         writer.writerow(
#             [
#                 "Strain-LSTM",
#             ]
#         )
#         writer.writerow(
#             [
#                 strain_lstm,
#             ]
#         )
#         writer.writerow(
#             [
#                 "SLSTM_mean_rel",
#                 "SLSTM_mean_rel_end",
#                 "LSTM_mean_rel",
#                 "LSTM_mean_rel_end",
#             ]
#         )
#         writer.writerow(
#             [
#                 slstm_test,
#                 slstm_test_end,
#                 lstm_test,
#                 lstm_test_end,
#             ]
#         )
#
#     return None


def main(device):
    """Main function for the elasticity experiment."""
    ELASTIC_MODULUS = 2.1e5
    BATCH_SIZE = 1024
    NUM_SAMPLES_TRAIN = BATCH_SIZE * 1
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 2000

    convergence(
        device=device,
        epochs=EPOCHS,
        elastic_modulus=ELASTIC_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples_train=NUM_SAMPLES_TRAIN,
        num_samples_val=NUM_SAMPLES_VAL,
        num_samples_test=NUM_SAMPLES_TEST,
        hidden=128,
    )

    # comparison(
    #     device=device,
    #     data_train=data_train,
    #     data_val=data_val,
    #     data_test=data_test,
    #     data_predict=data_predict,
    #     timesteps=TIMESTEPS,
    #     epochs=EPOCHS,
    #     mean_strain=mean_strain,
    #     std_strain=std_strain,
    #     mean_stress=mean_stress,
    #     std_stress=std_stress,
    # )

    return None


if __name__ == "__main__":
    main(device="cuda")
