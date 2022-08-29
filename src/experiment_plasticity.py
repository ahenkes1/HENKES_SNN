"""Plasticity using SNN. Section ...."""
import dataset
import model


def main():
    """Main function for the plasticity experiment."""
    YIELD_STRESS = 300
    ELASTIC_MODULUS = (2.1e5,)
    HARDENING_MODULUS = (2.1e5 / 100,)
    BATCH_SIZE = 32
    TIMESTEPS = 100
    NUM_SAMPLES_TRAIN = int(1e3)
    NUM_SAMPLES_VAL = int(1e3)
    NUM_SAMPLES_TEST = int(1e3)
    NUM_HIDDEN = 128

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

    slstm = model.SLSTM(timesteps=TIMESTEPS, hidden=NUM_HIDDEN)

    return None


if __name__ == "__main__":
    main()
