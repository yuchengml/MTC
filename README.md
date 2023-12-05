<a name="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">MTC</h3>

  <p align="center">
    Implementation of a multi-task model for encrypted network traffic classification based on transformer and 1D-CNN.
    <br />
    <a href="https://www.techscience.com/iasc/v37n1/52667/html">[Paper]</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#features">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#experiments">Experiments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This is a development project based on an existing work. Following the model architecture, parameters, and utilizing some datasets mentioned in the original paper, the goal is to implement the experimental results. However, performance matching the paper cannot be guaranteed. For simplicity in development, this project exclusively utilizes the ISCX VPN-non VPN dataset and follows the preprocessing methods outlined in the paper. Finally, the model is trained according to the parameters specified in the paper.

### Built With
[![PyTorch][pytorch-shield]][pytorch-url]
[![W&B][wandb-shield]][wandb-url]

<!-- GETTING STARTED -->
## Getting Started
Build the Python environment on either cloud or on-premises machines. 
To facilitate model training, please follow these simple example steps.

### Prerequisites

* Install python packages
  ```sh
  pip install requirements.txt
  ```
* Download ISCX VPN-non VPN dataset from [here](https://www.unb.ca/cic/datasets/vpn.html).

<!-- USAGE EXAMPLES -->
## Usage
* Preprocess ISCX VPN-non VPN data through a `Makefile`
  * Several pickle files would be generated in `data/`
    ```shell
    make preprocess
    ```

* To execute various model training tasks through a `Makefile`, follow the step below:
  * Execute all training jobs for each model
    ```shell
    make train-all
    ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FEATURES -->
## Features
- Multi-task model Implementation
  - [x] 1D-CNN model
  - [x] Transformer  model
  - [x] MTC 
    - 1D-CNN + transformer + fusion blocks

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [K. Wang, J. Gao and X. Lei, "Mtc: a multi-task model for encrypted network traffic classification based on transformer and 1d-cnn," Intelligent Automation & Soft Computing, vol. 37, no.1, pp. 619–638, 2023.
](https://www.techscience.com/iasc/v37n1/52667/html)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/yuchengml/MTC/blob/main/LICENSE
[pytorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[wandb-shield]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white
[wandb-url]: https://wandb.ai/site

<!-- EXPERIMENTS -->
## Experiments
### Dataset info
- ISCX VPE-non VPN
  - After filtering by application type

    | Application  | Amount    | 
    | ------------ | --------- |
    | aim_chat     | 2366      |
    | email        | 19705     |
    | facebook     | 2472071   |
    | ftps         | 4378      |
    | gmail        | 5242      |
    | hangouts     | 4419276   |
    | icq          | 2490      |
    | netflix      | 474       |
    | scp          | 19270     |
    | sftp         | 1351      |
    | skype        | 3727604   |
    | spotify      | 939       |
    | torrent      | 6885      |
    | vimeo        | 611       |
    | voipbuster   | 1559956   |
    | youtube      | 2230      |

