# Google Compute HELP Page

For Google Cloud ssh sessions to remain open:

```bash
sudo /sbin/sysctl -w net.ipv4.tcp_keepalive_time=60 net.ipv4.tcp_keepalive_intvl=60 net.ipv4.tcp_keepalive_probes=5
sudo echo -e 'net.ipv4.tcp_keepalive_time=60\nnet.ipv4.tcp_keepalive_intvl=60\nnet.ipv4.tcp_keepalive_probes=5' >> /etc/sysctl.conf
```

## Using Google Cloud with CUDA

Installation instructions for CUDA for Google Cloud: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#os_support.

**For Google Compute, use a Ubuntu instance with GPUs. Install Cuda using the instructions at**
https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#secure-boot

Following commands for installing a nvidia driver with secure boot didn't work, but may be helpful to have the commands.
```sh
    1  nvcc --version
    2  sudo nvidia-smi
    3  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    4  python
    5  ls
    6  ./driver_installer.run --help
    7  ./driver_installer.run -A
    8  sudo ./driver_installer.run --dkms -a --no-drm --install-libglvnd
    9  ls
   10  sudo nvidia-smi 
   11  nvcc --version
   12  nvidia-smi 
   13  python
   14  lsb_release -a
   15  lspci | grep -i nvidia
   16  uname -m && cat /etc/*release
   17  sudo apt install pkg-config 
   18  sudo ./driver_installer.run --dkms -a --no-drm --install-libglvndopenssl req -new -x509 -newkey rsa:2048 -keyout Nvidia.key -outform DER -out Nvidia.der -nodes -days 36500 -subj "/CN=Graphics Drivers"
   19  openssl req -new -x509 -newkey rsa:2048 -keyout Nvidia.key -outform DER -out Nvidia.der -nodes -days 36500 -subj "/CN=Graphics Drivers"
   20  ls
   21  sudo mokutil --import Nvidia.der 
   22  sudo ./driver_installer.run --dkms -a -s --no-drm --install-libglvnd --module-signing-secret-key=~/Nvidia.key --module-signing-public-key=~/Nvidia.der
   23  cat /var/log/nvidia-installer.log 
   24  ls
   25  pwd
   26  sudo ./driver_installer.run --dkms -a -s --no-drm --install-libglvnd --module-signing-secret-key=/home/ryan_mccoppin_1_ctr_afrl_af_mil/Nvidia.key --module-signing-public-key=/home/ryan_mccoppin_1_ctr_afrl_af_mil/Nvidia.der
   27  sudo nvidia-smi
```

### Not recommended to use Debian

If you are using Debian...
Install cuda from the NVIDIA documentation: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network

The following still fails to find the driver.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```
