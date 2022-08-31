"""Plasticity using SNN. Section ...."""
import dataset
import model
import trainer


def main(device):
    """Main function for the plasticity experiment."""
    YIELD_STRESS = 300
    ELASTIC_MODULUS = 2.1e5
    HARDENING_MODULUS = 2.1e5 / 100
    BATCH_SIZE = 1024
    TIMESTEPS = 100
    NUM_SAMPLES_TRAIN = int(1e4)
    NUM_SAMPLES_VAL = BATCH_SIZE
    NUM_SAMPLES_TEST = NUM_SAMPLES_VAL
    NUM_HIDDEN = 512
    EPOCHS = NUM_SAMPLES_TRAIN // BATCH_SIZE * 100

    data_train = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_TRAIN,
        timesteps=TIMESTEPS,
    )
    data_val = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_VAL,
        timesteps=TIMESTEPS,
    )
    data_test = dataset.plasticity(
        yield_stress=YIELD_STRESS,
        elastic_modulus=ELASTIC_MODULUS,
        hardening_modulus=HARDENING_MODULUS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_TEST,
        timesteps=TIMESTEPS,
    )

    slstm = model.SLSTM(timesteps=TIMESTEPS, hidden=NUM_HIDDEN).to(
        device=device
    )

    training_hist = trainer.training(
        dataloader_train=data_train,
        dataloader_val=data_val,
        model=slstm,
        learning_rate=1e-3,
        optimizer="adamw",
        device=device,
        epochs=EPOCHS,
    )

    testing_results = trainer.test(
        dataloader_test=data_test,
        model=slstm,
        device=device,
    )

    prediction = trainer.predict(
        dataloader_test=data_test,
        model=slstm,
        device=device,
        timesteps=TIMESTEPS,
    )

    return None


if __name__ == "__main__":
    main(device="cpu")
