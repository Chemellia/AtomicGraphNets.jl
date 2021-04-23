# Contribution Guide

We very happily accept contributions from the community to make our packages better! For the smoothest experience, please read this document and follow the guidelines and we can hopefully get your PR merged in a jiffy! We've tried to keep the guidelines lightweight, reasonable, and not too onerous. :)

(Don't know what a PR is? Know how to write Julia code but never contributed to a package before? Refer to the [Getting Started](#getting-started) section further on down the page.)

Thanks to the [OpenMM contribution guide](https://github.com/openmm/openmm/blob/master/CONTRIBUTING.md) and the [SciML ColPrac document](http://colprac.sciml.ai), which were the main inspirations/starting points for the suggestions contained herein.

## Guidelines

* In general, unless a change is very minor (e.g. fixing a typo), open an issue before opening a pull request that fixes that issue. This allows open discussion, collaboration, and prioritization of changes to the code. Please also label the issue appropriately. We use a set of labels that is slightly expanded from the [GitHub standard set](https://docs.github.com/en/github/managing-your-work-on-github/managing-labels#about-default-labels):

| Label              | Description                                                                                                                    |
| -------------      | -------------                                                                                                                  |
| `breaking`         | Indicates a pull request that introduces breaking changes                                                                      |
| `bug`              | Indicates an unexpected problem or unintended behavior                                                                         |
| `documentation`    | Indicates a need for improvements or additions to documentation                                                                |
| `duplicate`        | Indicates similar issues or pull requests                                                                                      |
| `enhancement`      | Indicates new feature requests                                                                                                 |
| `good first issue` | Indicates a good issue for first-time contributors                                                                             |
| `help wanted`      | Indicates that a maintainer wants help on an issue or pull request                                                             |
| `invalid`          | Indicates that an issue or pull request is no longer relevant                                                                  |
| `longterm`         | Indicates a feature that we intend to implement, but is not high-priority right now (generally only to be used by maintainers) |
| `priority`         | Indicates an issue that is high-priority (generally only to be used by maintainers)                                            |
| `question`         | Indicates that an issue or pull request needs more information                                                                 |
| `wontfix`          | Indicates that work won't continue on an issue or pull request                                                                 |

* If you are adding/changing features, make sure to add/update tests (DO NOT comment out tests!) and documentation accordingly! Ideally, if relevant, include example usage.
* Keep things modular! If you are fixing/adding multiple things, do so via separate issues/PR's to streamline review and merging.
* TODO: add style guide notes

## Getting Started

Coming...
