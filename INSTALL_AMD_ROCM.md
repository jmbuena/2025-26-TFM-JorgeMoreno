List of commands to install PyTorch when using AMD ROCm:

Based on the [official documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html#option-a-pytorch-via-pip-installation):

## Installing the Wheels from AMD directly:
> It is recommended to use these wheels instead of the ones provided by PyTorch.

Download the wheels

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
```

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
```

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
```

```bash
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
```

```bash
pip install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
```

Then install them using:

```bash
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
cd ${location}/torch/lib/
rm libhsa-runtime64.so*
```

## Testing the installation

```bash
python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
```

```bash
python3 -c 'import torch; print(torch.cuda.is_available())'
```

```bash
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```
