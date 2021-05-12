# Contribution Guide

We very happily accept contributions from the community to make our packages better! For the smoothest experience, please read this document and follow the guidelines and we can hopefully get your PR merged in a jiffy! We've tried to keep the guidelines lightweight, reasonable, and not too onerous. :)

(Don't know what a PR is? Know how to write Julia code but never contributed to a package before? Refer to the [Getting Started](#getting-started) section further on down the page.)

Thanks to the [OpenMM contribution guide](https://github.com/openmm/openmm/blob/master/CONTRIBUTING.md) and the [SciML ColPrac document](http://colprac.sciml.ai), which were the main inspirations/starting points for the suggestions contained herein.

## Guidelines

* Commit frequently and make the commit messages detailed! Ideally specifying directory/file as well as nature of changes. A sample commit message format could be:
```
directory of file affected: changes introduced

...commit message explicitly stating the changes made. this should be concise, and crisp enough, that the maintainers must be able to understand the changes this commit introduces without having to go through the diff-logs... 

Signed-off/Co-authored with/Suggested-by messages for credit where it's due
```
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
| `performance`      | Indicates an issue/PR to improve code performance.                                                                             |
| `priority`         | Indicates an issue that is high-priority (generally only to be used by maintainers)                                            |
| `question`         | Indicates that an issue or pull request needs more information                                                                 |
| `wontfix`          | Indicates that work won't continue on an issue or pull request                                                                 |

* If you are adding/changing features, make sure to add/update tests (DO NOT comment out tests!) and documentation accordingly! Ideally, if relevant, include example usage.
* Keep things modular! If you are fixing/adding multiple things, do so via separate issues/PR's to streamline review and merging.
* It is recommended that you set up [JuliaFormatter](https://domluna.github.io/JuliaFormatter.jl/dev/) (also see [VSCode extension](https://marketplace.visualstudio.com/items?itemName=singularitti.vscode-julia-formatter)). A GitHub action will check that code adheres to the style guide.

## Getting Started

We welcome contributions of well-written code from folks with all levels of software engineering experience! There are a TON of great guides out there for all aspects of collaborative development, so rather than reinventing the wheel, here are a few starting points for common things folks need/want to learn:

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/): An awesome high-level introduction covering philosophy, communication, and some best practices, and including links to other more detailed resources.

* [`first-contributions`](https://github.com/firstcontributions/first-contributions): A GitHub project designed to walk beginners through making a first contribution!

* [Resources to learn Git](https://try.github.io) compiled by GitHub.

* Is something missing here? Open a PR to add it! :slightly_smiling_face:
