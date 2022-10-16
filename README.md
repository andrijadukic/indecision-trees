# indecision-trees

A python implementation of ID3 decision trees and random forest.
The main goal of this repository is to serve as a simple, clean python implementation for people trying to learn the
concepts.

### Running the CLI

The CLI requires three parameters to run:

- path to the train dataset (any csv dataset, where the last column acts as the class label)
- path to the test dataset (any csv dataset, where the last column acts as the class label)
- path to the .cfg file (see examples in the [config](config) folder)

Type ```python main.py -h``` to see the available options.