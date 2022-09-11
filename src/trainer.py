"""Trainer for the spiking neural model."""
import statistics
import torch
import tqdm


def training(
    dataloader_train,
    dataloader_val,
    model,
    learning_rate,
    device,
    epochs,
    optimizer,
    savepath,
):
    model.train(mode=True)
    # model = model.to(device)

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

    epoch_loss_train = []
    epoch_loss_val = []

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

            avg_batch_loss_train = (
                sum(loss_epoch_train) / minibatch_counter_train
            )
            avg_batch_loss_val = sum(loss_epoch_val) / minibatch_counter_val

            epoch_loss_train.append(avg_batch_loss_train)
            epoch_loss_val.append(avg_batch_loss_val)

            pbar.set_postfix(
                loss_train=avg_batch_loss_train,
                loss_val=avg_batch_loss_val,
            )

            if min_valid_loss > avg_batch_loss_val:
                torch.save(model.state_dict(), savepath)
            else:
                pass

    print(f"{79 * '='}")

    return {
        "loss_mse_train": loss_mse_train_lst,
        "rel_err_train": rel_err_train_lst,
        "rel_err_end_train": rel_err_end_train_lst,
        "loss_mse_val": loss_mse_val_lst,
        "rel_err_val": rel_err_val_lst,
        "rel_err_end_val": rel_err_end_val_lst,
        "epoch_loss_train": epoch_loss_train,
        "epoch_loss_val": epoch_loss_val,
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


def predict(
    model,
    device,
    dataloader_predict,
    mean_strain,
    std_strain,
    mean_stress,
    std_stress,
    num_samples,
):
    """Carry out predictions with model on data."""

    space = 20
    print(
        f"{79 * '='}\n"
        f"{' ':<{space}}{'Predicting':^39}{' ':>{space}}\n"
        f"{79 * '-'}"
    )

    test_batch = iter(dataloader_predict)
    strain_norm, stress_norm = next(test_batch)

    strain_norm = strain_norm[0:num_samples, :, :]
    stress_norm = stress_norm[0:num_samples, :, :]

    if len(strain_norm.shape) == 2:
        strain_norm = torch.unsqueeze(strain_norm, 0)
        stress_norm = torch.unsqueeze(stress_norm, 0)

    stress = (stress_norm * std_stress) + mean_stress

    with torch.no_grad():
        model.eval()

        feature = torch.swapaxes(input=strain_norm, axis0=0, axis1=1)
        label = torch.swapaxes(input=stress, axis0=0, axis1=1)

        feature = feature.to(device)
        label = label.to(device)

        out_dict = model(feature)
        mem = out_dict["membrane_potential"]

    prediction = (mem * std_stress) + mean_stress
    strain = (feature * std_strain) + mean_strain

    feature = torch.squeeze(strain).cpu().numpy()
    label = torch.squeeze(label).cpu().numpy()
    prediction = torch.squeeze(prediction).cpu().numpy()

    print(f"{79 * '='}")

    return {"strain": feature, "prediction": prediction, "true": label}


if __name__ == "__main__":
    pass
