# Example 2: QM9

In this example we will use the same network architecture but use data from the QM9 dataset. Note that the .xyz files provided within the QM9 dataset are not parsable directly by ASE, you need to remove the last couple lines, which is easy enough to script yourself, but I've included a small set of them here for demonstration purposes.

NB: the actual model performance on QM9 is not that great because we're currently not encoding a variety of important features for organic molecules. This is provided mainly to show the processing of a different dataset and demonstrate batch processing capabilities.