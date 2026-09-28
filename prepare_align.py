import hparams as hp

def main():
    """Alignment preparation entry point.

    The supplied KSS portfolio source already contains aligned/preprocessed
    artifacts. Dataset-specific alignment generation should be implemented
    separately when rebuilding the dataset from raw audio.
    """
    print(f"Dataset: {hp.dataset}")
    print(f"Data path: {hp.data_path}")

if __name__ == "__main__":
    main()
