"""Trainer for the spiking neural model."""
import statistics
import torch
import tqdm
import pprint


def training(
    dataloader_train,
    dataloader_val,
    model,
    learning_rate,
    device,
    epochs,
    optimizer,
):
    model.train(mode=True)

    optimizer = str(optimizer)
    if optimizer == "nadam":
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )

    elif optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )

    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

    else:
        raise NotImplementedError()

    loss_function = torch.nn.MSELoss()

    loss_mse_train_lst = []
    rel_err_train_lst = []
    rel_err_end_train_lst = []

    loss_mse_val_lst = []
    rel_err_val_lst = []
    rel_err_end_val_lst = []

    space = 20
    print(
        f"{79 * '='}\n"
        f"{' ':<20}{'Training':^39}{' ':>20}\n"
        f"{79 * '-'}\n"
        f"{'Optimizer:':<{space}}\n{optimizer}\n"
        f"{'Loss function:':<{space}}{loss_function}\n"
        f"{'Parameters:':<{space}}{device}\n"
    )

    min_valid_loss = torch.inf

    with tqdm.trange(int(epochs)) as pbar:
        for _ in pbar:
            minibatch_counter_train = 0
            loss_epoch_train = []
            train_batch = iter(dataloader_train)
            for feature, label in train_batch:
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
                out_dict = model(feature)
                mem = out_dict["membrane_potential"]

                rel_err_train = torch.linalg.norm(
                    (mem - label), dim=-1
                ) / torch.linalg.norm(label, dim=-1)
                rel_err_train = torch.mean(rel_err_train[1:, :])

                rel_err_end_train = torch.linalg.norm(
                    (mem[-1, :, :] - label[-1, :, :]), dim=-1
                ) / torch.linalg.norm(label[-1, :, :], dim=-1)
                rel_err_end_train = torch.mean(rel_err_end_train)

                loss_train = loss_function(mem, label)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                loss_mse_train_lst.append(loss_train.item())
                rel_err_train_lst.append(rel_err_train.item())
                rel_err_end_train_lst.append(rel_err_end_train.item())

                loss_epoch_train.append(loss_train.item())
                minibatch_counter_train += 1

                pbar.set_postfix(
                    l_train=loss_train.item(),
                    r_train=rel_err_train.item(),
                    e_train=rel_err_end_train.item(),
                    l_val=loss_val.item(),
                    r_val=rel_err_val.item(),
                    e_val=rel_err_end_val.item(),
                )

            minibatch_counter_val = 0
            loss_epoch_val = []
            val_batch = iter(dataloader_val)
            for feature, label in val_batch:
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
                out_dict = model(feature)
                mem = out_dict["membrane_potential"]

                rel_err_val = torch.linalg.norm(
                    (mem - label), dim=-1
                ) / torch.linalg.norm(label, dim=-1)
                rel_err_val = torch.mean(rel_err_val[1:, :])

                rel_err_end_val = torch.linalg.norm(
                    (mem[-1, :, :] - label[-1, :, :]), dim=-1
                ) / torch.linalg.norm(label[-1, :, :], dim=-1)
                rel_err_end_val = torch.mean(rel_err_end_val)

                loss_val = loss_function(mem, label)

                loss_mse_val_lst.append(loss_val.item())
                rel_err_val_lst.append(rel_err_val.item())
                rel_err_end_val_lst.append(rel_err_end_val.item())

                loss_epoch_val.append(loss_val.item())
                minibatch_counter_val += 1

                pbar.set_postfix(
                    l_train=loss_train.item(),
                    r_train=rel_err_train.item(),
                    e_train=rel_err_end_train.item(),
                    l_val=loss_val.item(),
                    r_val=rel_err_val.item(),
                    e_val=rel_err_end_val.item(),
                )

            avg_batch_loss_train = (
                sum(loss_epoch_train) / minibatch_counter_train
            )
            avg_batch_loss_val = sum(loss_epoch_val) / minibatch_counter_val

        if min_valid_loss > avg_batch_loss_val:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f\
            }--->{avg_batch_loss_train:.6f}) \t Saving The Model"
            )
            min_valid_loss = avg_batch_loss_val

            # Saving State Dict
            torch.save(model.state_dict(), "./saved_model/saved_model.pth")

    print(f"{79 * '='}")

    return {
        "loss_mse_train": loss_mse_train_lst,
        "rel_err_train": rel_err_train_lst,
        "rel_err_end_train": rel_err_end_train_lst,
        "loss_mse_val": loss_mse_val_lst,
        "rel_err_val": rel_err_val_lst,
        "rel_err_end_val": rel_err_end_val_lst,
    }


def test(model, device, dataloader_test):
    """Test performance on a test set."""
    loss_function_L1 = torch.nn.L1Loss()

    loss_test_L1 = []
    rel_err_test = []
    rel_err_end_test = []

    space = 20
    print(f"{79 * '='}\n" f"{' ':<20}{'Testing':^39}{' ':>20}\n" f"{79 * '-'}")

    with torch.no_grad():
        model.eval()

        test_batch = iter(dataloader_test)

        for feature, label in test_batch:
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)

            feature = feature.to(device)
            label = label.to(device)

            out_dict = model(feature)
            mem = out_dict["membrane_potential"]

            rel_err = torch.linalg.norm(
                (mem - label), dim=-1
            ) / torch.linalg.norm(label, dim=-1)
            rel_err = torch.mean(rel_err[1:, :])

            rel_err_end = torch.linalg.norm(
                (mem[-1, :, :] - label[-1, :, :]), dim=-1
            ) / torch.linalg.norm(label[-1, :, :], dim=-1)
            rel_err_end = torch.mean(rel_err_end)

            loss_val_L1 = loss_function_L1(mem, label)

            loss_test_L1.append(loss_val_L1.item())
            rel_err_test.append(rel_err.item())
            rel_err_end_test.append(rel_err_end.item())

        mean_L1 = statistics.mean(loss_test_L1)
        mean_rel = statistics.mean(rel_err_test)
        mean_rel_end = statistics.mean(rel_err_end_test)

    print(f"{'Mean L1-loss:':<{space}}{mean_L1:1.2e}")
    print(f"{'Mean rel. err:':<{space}}{mean_rel:1.2e}")
    print(f"{'Mean rel. err end:':<{space}}{mean_rel_end:1.2e}")
    print(f"{79 * '='}")

    return {
        "mean_loss_test_l1": mean_L1,
        "mean_rel_err_test": mean_rel,
        "mean_rel_err_end_test": mean_rel_end,
    }


def predict(model, strain, stress, device, timesteps):
    """Carry out predictions with model on data."""

    space = 20
    print(
        f"{79 * '='}\n" f"{' ':<20}{'Predicting':^39}{' ':>20}\n" f"{79 * '-'}"
    )

    with torch.no_grad():
        model.eval()

        strain = strain.to(device)
        stress = stress.to(device)

        out, _ = model(strain.view(timesteps, 1, 1))

    prediction = out
    abs_error = torch.linalg.norm(stress - prediction)

    stress = torch.squeeze(stress).cpu().numpy().tolist()
    prediction = torch.squeeze(prediction).cpu().numpy().tolist()

    for idx, element in enumerate(stress):
        stress[idx] = "{:1.4e}".format(element)

    for idx, element in enumerate(prediction):
        prediction[idx] = "{:1.4e}".format(element)

    print(
        f"{'L1 error:':<{space}}{abs_error:1.2e}\n\n"
        f"{'True':<{16}}{'Prediction':<{16}}"
    )
    pprint.pprint(list(zip(stress, prediction)))
    print(f"{79 * '='}")

    return out


if __name__ == "__main__":
    pass