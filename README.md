
# [hp-VPINNs: Variational Physics Informed Neural Networks With Domain Decomposition]


We introduce the variational physics informed neural networks – a general framework to solve differential equations.

For more information, please refer to the following: 

  - Kharazmi, Ehsan,  Zhongqiang Zhang, and George E. Karniadakis. "[hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition](https://arxiv.org/abs/2003.05385)." arXiv preprint arXiv:2003.05385 (2020).

  - Kharazmi, Ehsan,  Zhongqiang Zhang, and George E. Karniadakis. "[Variational Physics-Informed Neural Networks For Solving Partial Differential Equations](https://arxiv.org/abs/1912.00873)." arXiv preprint arXiv:1912.00873 (2019).


## Citation

  @article{kharazmi2020hp,
    title={hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition},
    author={Kharazmi, Ehsan and Zhang, Zhongqiang and Karniadakis, George Em},
    journal={arXiv preprint arXiv:2003.05385},
    year={2020}
  }

  @article{kharazmi2019variational,
    title={Variational physics-informed neural networks for solving partial differential equations},
    author={Kharazmi, Ehsan and Zhang, Zhongqiang and Karniadakis, George Em},
    journal={arXiv preprint arXiv:1912.00873},
    year={2019}
  }


## Installation and usage 

Clone the code to you repo 

### Setup a Venv

Navigate into the directory using `cd` and then create a `venv`

```
cd VPINNS
python3 -m venv .
```

Source the venv

```
source bin/activate
```

### Running the code

Then run the code using

```
python3 main/SingularlyPerturbed/hp-VPINN-Singularly_perturbed_2D.py input.yaml
```
